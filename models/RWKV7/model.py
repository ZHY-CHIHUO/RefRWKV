########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, math, gc, importlib
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy
import copy

if importlib.util.find_spec("deepspeed"):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam


env_defaults = {
    "RWKV_JIT_ON": "0",
    "RWKV_HEAD_SIZE": "64",
    "RWKV_FLOAT_MODE": "bf16",
    "RWKV_MY_TESTING": "x070",
}
for key, val in env_defaults.items():
    if key not in os.environ:
        os.environ[key] = val

try:
    print("RWKV_MY_TESTING", os.environ["RWKV_MY_TESTING"])
except:
    os.environ["RWKV_MY_TESTING"] = ""


def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop
if os.environ["RWKV_JIT_ON"] == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method


########################################################################################################
# CUDA Kernel
########################################################################################################

from torch.utils.cpp_extension import load

HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE"])

if "x070" in os.environ["RWKV_MY_TESTING"]:
    CHUNK_LEN = 16
    assert HEAD_SIZE == 64  # can change 64 to your HEAD_SIZE

    # check https://github.com/BlinkDL/RWKV-CUDA/blob/main/rwkv7_fast_fused/rwkv7_cuda_benchmark.py

    flags = [
        "-res-usage",
        f"-D_N_={HEAD_SIZE}",
        f"-D_CHUNK_LEN_={CHUNK_LEN}",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        "--extra-device-vectorization",
    ]
    load(
        name="rwkv7_clampw",
        sources=[f"cuda/rwkv7_clampw.cu", "cuda/rwkv7_clampw.cpp"],
        is_python_module=False,
        verbose=True,
        extra_cuda_cflags=flags,
    )

    class RWKV7_CLAMPW_CUDA_OP(torch.autograd.Function):
        @staticmethod
        def forward(ctx, r, w, k, v, a, b):
            B, T, H, N = r.shape
            assert (
                T % CHUNK_LEN == 0
            )  # if T%CHUNK_LEN != 0: pad your input to T%CHUNK_LEN == 0, or change CHUNK_LEN (will be slower)
            assert all(i.dtype == torch.bfloat16 for i in [r, w, k, v, a, b])
            assert all(i.is_contiguous() for i in [r, w, k, v, a, b])
            y = torch.empty_like(v)
            s = torch.empty(
                B, H, T // CHUNK_LEN, N, N, dtype=torch.float32, device=w.device
            )
            sa = torch.empty(B, T, H, N, dtype=torch.float32, device=w.device)
            torch.ops.rwkv7_clampw.forward(r, w, k, v, a, b, y, s, sa)
            ctx.save_for_backward(r, w, k, v, a, b, s, sa)
            return y

        @staticmethod
        def backward(ctx, dy):
            assert all(i.dtype == torch.bfloat16 for i in [dy])
            assert all(i.is_contiguous() for i in [dy])
            r, w, k, v, a, b, s, sa = ctx.saved_tensors
            dr, dw, dk, dv, da, db = [torch.empty_like(x) for x in [r, w, k, v, a, b]]
            torch.ops.rwkv7_clampw.backward(
                r, w, k, v, a, b, dy, s, sa, dr, dw, dk, dv, da, db
            )
            return dr, dw, dk, dv, da, db

    def RWKV7_CLAMPW_CUDA(r, w, k, v, a, b):
        B, T, HN = r.shape
        r, w, k, v, a, b = [
            i.view(B, T, HN // 64, 64) for i in [r, w, k, v, a, b]
        ]  # can change 64 to your HEAD_SIZE. have to hard-code the number here, or pytorch will complain
        return RWKV7_CLAMPW_CUDA_OP.apply(r, w, k, v, a, b).view(B, T, HN)


########################################################################################################


class RWKV_Tmix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.my_testing = args.my_testing

        self.head_size = args.head_size
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, C)
            for i in range(C):
                ddd[0, 0, i] = i / C

            self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

            def ortho_init(x, scale):
                with torch.no_grad():
                    shape = x.shape
                    if len(shape) == 2:
                        gain = (
                            math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
                        )
                        nn.init.orthogonal_(x, gain=gain * scale)
                    elif len(shape) == 3:
                        gain = (
                            math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
                        )
                        for i in range(shape[0]):
                            nn.init.orthogonal_(x[i], gain=gain * scale)
                    else:
                        assert False
                    return x

            www = torch.zeros(C)
            zigzag = torch.zeros(C)
            linear = torch.zeros(C)
            for n in range(C):
                linear[n] = n / (C - 1) - 0.5
                zigzag[n] = ((n % N) - ((N - 1) / 2)) / ((N - 1) / 2)
                zigzag[n] = zigzag[n] * abs(zigzag[n])
                www[n] = -6 + 6 * (n / (C - 1)) ** (1 + 1 * ratio_0_to_1**0.3)

            D_DECAY_LORA = max(32, int(round((2.5 * (C**0.5)) / 32) * 32))  # suggestion
            self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
            self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
            self.w0 = nn.Parameter(www.reshape(1, 1, C) + 0.5 + zigzag * 2.5)

            D_AAA_LORA = max(32, int(round((2.5 * (C**0.5)) / 32) * 32))  # suggestion
            self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
            self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
            self.a0 = nn.Parameter(
                torch.zeros(1, 1, C) - 0.19 + zigzag * 0.3 + linear * 0.4
            )

            D_MV_LORA = max(32, int(round((1.7 * (C**0.5)) / 32) * 32))  # suggestion
            self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
            self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
            self.v0 = nn.Parameter(torch.zeros(1, 1, C) + 0.73 - linear * 0.4)

            # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
            D_GATE_LORA = max(32, int(round((5 * (C**0.5)) / 32) * 32))  # suggestion
            self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
            self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

            self.k_k = nn.Parameter(torch.zeros(1, 1, C) + 0.71 - linear * 0.1)
            self.k_a = nn.Parameter(torch.zeros(1, 1, C) + 1.02)
            self.r_k = nn.Parameter(torch.zeros(H, N) - 0.04)

            self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
            self.receptance = nn.Linear(C, C, bias=False)
            self.key = nn.Linear(C, C, bias=False)
            self.value = nn.Linear(C, C, bias=False)
            self.output = nn.Linear(C, C, bias=False)
            self.ln_x = nn.GroupNorm(H, C, eps=64e-5)  # !!! notice eps value !!!

            self.receptance.weight.data.uniform_(-0.5 / (C**0.5), 0.5 / (C**0.5))
            self.key.weight.data.uniform_(-0.05 / (C**0.5), 0.05 / (C**0.5))
            self.value.weight.data.uniform_(-0.5 / (C**0.5), 0.5 / (C**0.5))
            self.output.weight.data.zero_()

    @MyFunction
    def forward(self, x, v_first):
        B, T, C = x.size()
        H = self.n_head
        xx = self.time_shift(x) - x

        xr = x + xx * self.x_r
        xw = x + xx * self.x_w
        xk = x + xx * self.x_k
        xv = x + xx * self.x_v
        xa = x + xx * self.x_a
        xg = x + xx * self.x_g

        r = self.receptance(xr)
        w = (
            self.w0 + torch.tanh(xw @ self.w1) @ self.w2
        )  # will be soft-clamped to (-inf, -0.5) and exp(-exp(w)) in RWKV7_CLAMPW_CUDA kernel
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v  # store the v of the first layer
        else:
            v = v + (v_first - v) * torch.sigmoid(
                self.v0 + (xv @ self.v1) @ self.v2
            )  # add value residual
        a = torch.sigmoid(
            self.a0 + (xa @ self.a1) @ self.a2
        )  # a is "in-context learning rate"
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = k * self.k_k
        kk = F.normalize(kk.view(B, T, H, -1), dim=-1, p=2.0).view(B, T, C)
        k = k * (1 + (a - 1) * self.k_a)

        x = RWKV7_CLAMPW_CUDA(r, w, k, v, -kk, kk * a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + (
            (r.view(B, T, H, -1) * k.view(B, T, H, -1) * self.r_k).sum(
                dim=-1, keepdim=True
            )
            * v.view(B, T, H, -1)
        ).view(B, T, C)
        x = self.output(x * g)
        return x, v_first


########################################################################################################


class RWKV_CMix_x070(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = nn.Linear(args.n_embd, args.n_embd * 4, bias=False)
        self.value = nn.Linear(args.n_embd * 4, args.n_embd, bias=False)

        self.key.weight.data.uniform_(
            -0.5 / (args.n_embd**0.5), 0.5 / (args.n_embd**0.5)
        )
        self.value.weight.data.zero_()

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x) - x

        k = x + xx * self.x_k
        k = torch.relu(self.key(k)) ** 2

        return self.value(k)


########################################################################################################
# The RWKV Model with our blocks
########################################################################################################


class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)

    def forward(self, x, v_first):
        if self.layer_id == 0:
            x = self.ln0(x)

        x_attn, v_first = self.att(self.ln1(x), v_first)
        x = x + x_attn

        x = x + self.ffn(self.ln2(x))
        return x, v_first
    
class RWKV(nn.Module):
    """RWKV 的主体：多个 Block 堆叠 + 最后的 LayerNorm"""

    def __init__(self, args):
        super().__init__()
        self.args = args
        if not hasattr(args, 'dim_att'):
            args.dim_att = args.n_embd
        if not hasattr(args, 'dim_ffn'):
            args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32) # default = 3.5x emb size            
        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0

        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)

    def forward(self, x):
        v_first = torch.zeros_like(x)
        for block in self.blocks:
            if self.args.grad_cp == 1:
                x, v_first = deepspeed.checkpointing.checkpoint(block, x, v_first)
            else:
                x, v_first = block(x, v_first)
        x = self.ln_out(x)
        return x

class ChannelBiRWKV(nn.Module):
    """通道维度的双向交叉 RWKV（通道间全局建模）"""
    def __init__(self, channel_rwkv_args):
        super().__init__()
        self.n_embd = channel_rwkv_args.n_embd
        self.embed = nn.Linear(1, self.n_embd)
        self.rwkv = RWKV(channel_rwkv_args)   # 共享权重，用于正向+反向
        self.head = nn.Linear(self.n_embd, 1)
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).reshape(B * H * W, C)
        x_seq = x_flat.unsqueeze(-1)                     # (N, C, 1)
        x_seq = self.embed(x_seq)                        # (N, C, n_embd)

        pad_len = (16 - C % 16) % 16
        if pad_len > 0:
            x_seq = F.pad(x_seq, (0, 0, 0, pad_len))

        # 双向交叉
        out1 = self.rwkv(x_seq)
        out1_rev = out1.flip(1)
        out2_rev = self.rwkv(out1_rev)
        out2 = out2_rev.flip(1)

        if pad_len > 0:
            out2 = out2[:, :C, :]

        x_out = self.head(out2).squeeze(-1)
        x_out = x_out.view(B, H, W, C).permute(0, 3, 1, 2)
        return x + self.alpha * x_out

class SpatialBiRWKV(nn.Module):
    """空间维度的双向交叉 RWKV（空间位置间全局建模）"""
    def __init__(self, channels, spatial_rwkv_args):
        super().__init__()
        self.d_model = spatial_rwkv_args.n_embd
        self.proj_in = nn.Linear(channels, self.d_model)
        self.rwkv = RWKV(spatial_rwkv_args)   # 共享权重
        self.proj_out = nn.Linear(self.d_model, channels)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)       # (B, N, C)
        x = self.proj_in(x)                    # (B, N, d_model)

        # 双向交叉
        out1 = self.rwkv(x)
        out1_rev = out1.flip(1)
        out2_rev = self.rwkv(out1_rev)
        out2 = out2_rev.flip(1)

        x = self.proj_out(out2)
        x = x.transpose(1, 2).view(B, C, H, W)
        return x

########################################################################################################
# RIRBlock 与 U-Net 编码器/解码器
########################################################################################################

class BasicBlock(nn.Module):
    """基础残差块 (无BN) + 可选通道注意力"""
    def __init__(self, channels, use_ca=False):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.use_ca = use_ca
        if use_ca:
            self.ca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels // 16, 1),
                nn.ReLU(),
                nn.Conv2d(channels // 16, channels, 1),
                nn.Sigmoid()
            )

    def forward(self, x):
        res = self.conv2(self.relu(self.conv1(x)))
        if self.use_ca:
            res = res * self.ca(res)
        return x + res


class ResidualGroup(nn.Module):
    """一组 BasicBlock + 组内跳跃连接"""
    def __init__(self, channels, n_blocks=4, use_ca=False):
        super().__init__()
        self.blocks = nn.Sequential(*[BasicBlock(channels, use_ca) for _ in range(n_blocks)])

    def forward(self, x):
        return x + self.blocks(x)


class RIRBlock(nn.Module):
    def __init__(self, channels, n_groups=3, n_blocks=4,
                 use_ca=False, use_channel_rwkv=False, channel_rwkv_args=None):
        super().__init__()
        self.entry = nn.Conv2d(channels, channels, 3, padding=1)
        self.groups = nn.ModuleList([ResidualGroup(channels, n_blocks, use_ca)for _ in range(n_groups)])
        self.exit = nn.Conv2d(channels, channels, 3, padding=1)

        self.use_channel_rwkv = use_channel_rwkv
        if use_channel_rwkv:
            self.channel_rwkv = ChannelBiRWKV(channel_rwkv_args)

    def forward(self, x):
        res = self.entry(x)
        for group in self.groups:
            res = group(res)
        res = self.exit(res)
        if self.use_channel_rwkv:
            res = self.channel_rwkv(res)   # 通道全局调制
        return x + res


class DownBlock(nn.Module):
    """下采样 + RIRBlock"""
    def __init__(self, in_ch, out_ch, n_groups, n_blocks, use_ca=False, use_channel_rwkv=False, channel_rwkv_args=None):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.rir = RIRBlock(out_ch, n_groups, n_blocks, use_ca, use_channel_rwkv, channel_rwkv_args)

    def forward(self, x):
        return self.rir(self.conv(x))


class UpBlock(nn.Module):
    """上采样 + 跳跃连接 + 参考特征拼接 + RIRBlock"""
    def __init__(self, in_ch, skip_ch, ref_ch, out_ch, n_groups, n_blocks,
                 use_ca=False, use_channel_rwkv=False, channel_rwkv_args=None):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.fuse = nn.Conv2d(out_ch + skip_ch + ref_ch, out_ch, 1)
        self.rir = RIRBlock(out_ch, n_groups, n_blocks, use_ca, use_channel_rwkv, channel_rwkv_args)

    def forward(self, x, skip, ref):
        x = self.up(x)
        x = torch.cat([x, skip, ref], dim=1)
        x = self.fuse(x)
        return self.rir(x)

class RWKVSR(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.lr_size = args.lr_size
        self.hr_size = args.hr_size

        # 参数解耦：通道 RWKV 和空间 RWKV 使用独立配置
        self.channel_rwkv_args = args.channel_rwkv_args
        self.spatial_rwkv_args = args.spatial_rwkv_args

        # ===== 投影层 =====
        # 将拼接后的 lr1 + lr2 (共16通道) 投影到 4 通道
        self.lr_proj = nn.Conv2d(16, 4, 1)
        self.base_proj = nn.Conv2d(8, 4, 1)

        # ===== 目标编码器（带通道双向 RWKV）=====
        self.enc1 = DownBlock(4, 32, n_groups=3, n_blocks=4,
                              use_channel_rwkv=True, channel_rwkv_args=self.channel_rwkv_args)
        self.enc2 = DownBlock(32, 64, n_groups=3, n_blocks=4,
                              use_channel_rwkv=True, channel_rwkv_args=self.channel_rwkv_args)

        # ===== 参考编码器（轻量，无通道 RWKV）=====
        # ref_enc1: 256 -> 32  (3次下采样)
        self.ref_enc1 = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2, padding=1),  # 128
            nn.ReLU(inplace=True),
            RIRBlock(32, n_groups=2, n_blocks=4, use_ca=False),  # 轻量精炼
            nn.Conv2d(32, 32, 3, stride=2, padding=1),  # 64
            nn.ReLU(inplace=True),
            RIRBlock(32, n_groups=2, n_blocks=4, use_ca=False),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),  # 32
            nn.ReLU(inplace=True),
            RIRBlock(32, n_groups=2, n_blocks=4, use_ca=False),
        )
        # ref_enc2: 32 -> 16 (1次下采样)
        self.ref_enc2 = DownBlock(32, 64, n_groups=2, n_blocks=4, use_ca=False)

        # ===== 瓶颈：空间双向 RWKV =====
        self.bottleneck = SpatialBiRWKV(64, self.spatial_rwkv_args)

        # ===== 解码器 =====
        # dec2: 8 -> 16, 接收 e1(32ch) 和 ref2(64ch)
        self.dec2 = UpBlock(in_ch=64, skip_ch=32, ref_ch=64, out_ch=64,
                            n_groups=3, n_blocks=4,
                            use_channel_rwkv=True, channel_rwkv_args=self.channel_rwkv_args)
        # dec1: 16 -> 32, 接收 lr_fused(4ch) 和 ref1(32ch)
        self.dec1 = UpBlock(in_ch=64, skip_ch=4, ref_ch=32, out_ch=32,
                            n_groups=3, n_blocks=4,
                            use_channel_rwkv=True, channel_rwkv_args=self.channel_rwkv_args)

        # ===== 输出头 =====
        # 先将 32x32 特征上采样到 64x64，再卷积输出 4 通道残差
        self.dec0 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, 3, padding=1)
        )

        # 测试指标（可选）
        from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
        from torchmetrics import MeanSquaredError
        self.test_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.test_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.test_rmse = MeanSquaredError(squared=False)

    def forward(self, lr1, lr2, hr1):
        # 拼接低分影像并投影
        lr_cat = torch.cat([lr1, lr2], dim=1)          # (B, 16, 32, 32)
        lr_fused = self.lr_proj(lr_cat)                # (B, 4, 32, 32)

        # 目标编码器
        e1 = self.enc1(lr_fused)                       # (B, 32, 16, 16)
        e2 = self.enc2(e1)                             # (B, 64, 8, 8)

        # 参考编码器
        ref1 = self.ref_enc1(hr1)                      # (B, 32, 32, 32)
        ref2 = self.ref_enc2(ref1)                     # (B, 64, 16, 16)

        # 瓶颈
        b = self.bottleneck(e2)                        # (B, 64, 8, 8)

        # 解码器（融合跳跃连接和参考特征）
        d2 = self.dec2(b, e1, ref2)                    # (B, 64, 16, 16)
        d1 = self.dec1(d2, lr_fused, ref1)             # (B, 32, 32, 32)

        # 输出残差
        d0 = F.interpolate(d1, scale_factor=2, mode='bilinear', align_corners=False)
        res_64 = self.dec0(d0)                         # (B, 4, 64, 64)
        res = F.interpolate(res_64, scale_factor=4, mode='bilinear', align_corners=False)
        return res

    # ---------- 训练 / 验证 / 测试步骤保持不变 ----------
    def training_step(self, batch, batch_idx):
        lr1, hr1, lr2, hr2 = batch
        base_lr2 = self.base_proj(lr2)          # (B, 4, 32, 32)
        base = F.interpolate(base_lr2, scale_factor=8, mode='bicubic', align_corners=False)  # (B, 4, 256, 256)
        res = self(lr1, lr2, hr1)
        sr = base + res
        loss = F.l1_loss(sr, hr2)
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        lr1, hr1, lr2, hr2 = batch
        base_lr2 = self.base_proj(lr2)          # (B, 4, 32, 32)
        base = F.interpolate(base_lr2, scale_factor=8, mode='bicubic', align_corners=False)  # (B, 4, 256, 256)
        res = self(lr1, lr2, hr1)
        sr = base + res
        loss = F.l1_loss(sr, hr2)
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        lr1, hr1, lr2, hr2 = batch
        base_lr2 = self.base_proj(lr2)          # (B, 4, 32, 32)
        base = F.interpolate(base_lr2, scale_factor=8, mode='bicubic', align_corners=False)  # (B, 4, 256, 256)
        res = self(lr1, lr2, hr1)
        sr = base + res
        loss = F.l1_loss(sr, hr2)
        self.log("test_loss", loss, on_step=True, on_epoch=True)
        sr, hr2 = torch.clamp(sr, 0, 1), torch.clamp(hr2, 0, 1)
        self.test_psnr.update(sr, hr2)
        self.test_ssim.update(sr, hr2)
        self.test_rmse.update(sr, hr2)
        return loss

    def on_test_epoch_end(self):
        avg_psnr = self.test_psnr.compute()
        avg_ssim = self.test_ssim.compute()
        avg_rmse = self.test_rmse.compute()
        self.log("test_psnr_epoch", avg_psnr, sync_dist=True)
        self.log("test_ssim_epoch", avg_ssim, sync_dist=True)
        self.log("test_rmse_epoch", avg_rmse, sync_dist=True)
        self.test_psnr.reset()
        self.test_ssim.reset()
        self.test_rmse.reset()
        print(f"\nTest Results - PSNR: {avg_psnr:.4f} dB, SSIM: {avg_ssim:.4f}, RMSE: {avg_rmse:.6f}")

    # ---------- 优化器 ----------
    def configure_optimizers(self):
        args = self.args
        lr_decay, lr_1x, lr_2x = set(), set(), set()
        for n, p in self.named_parameters():
            if "att.w0" in n:
                lr_2x.add(n)
            elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0) and ".weight" in n:
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        param_dict = {n: p for n, p in self.named_parameters()}
        groups = [
            {"params": [param_dict[n] for n in lr_1x], "lr": args.lr_init, "weight_decay": 0.0},
            {"params": [param_dict[n] for n in lr_2x], "lr": args.lr_init * 2, "weight_decay": 0.0},
        ]
        if args.weight_decay > 0:
            groups.append({"params": [param_dict[n] for n in lr_decay], "lr": args.lr_init, "weight_decay": args.weight_decay})

        optimizer = torch.optim.AdamW(groups, betas=args.betas, eps=args.adam_eps, amsgrad=False)
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=1, threshold=1e-3, min_lr=5e-5)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss", "interval": "epoch", "frequency": 1}}

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False
