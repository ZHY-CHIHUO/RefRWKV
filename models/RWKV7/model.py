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


class RWKV(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.args = args
        if not hasattr(args, "dim_att"):
            args.dim_att = args.n_embd
        if not hasattr(args, "dim_ffn"):
            args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)

        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0

        # ---------- 图像参数 ----------
        self.hr_size = args.hr_size
        self.lr_size = args.lr_size
        self.patch_size_hr = args.patch_size_hr
        self.patch_size_lr = args.patch_size_lr
        self.hr_grid_size = self.hr_size // self.patch_size_hr
        self.lr_grid_size = self.lr_size // self.patch_size_lr
        self.hr_patches = self.hr_grid_size ** 2
        self.lr_patches = self.lr_grid_size ** 2

        self.hr_dim = 4 * self.patch_size_hr**2
        self.lr_dim = 8 * self.patch_size_lr**2

        # 动态计算序列长度（lr1 + hr1 + lr2 + hr2）
        self.total_seq_len = 2 * self.lr_patches + 2 * self.hr_patches

        # hr2 第一个 token 在整个序列中的起始索引（0-based）
        self.hr2_start_idx = 2 * self.lr_patches + self.hr_patches

        # ---------- Patch 投影 ----------
        self.hr_proj = nn.Linear(self.hr_dim, args.n_embd)
        self.lr_proj = nn.Linear(self.lr_dim, args.n_embd)

        # RWKV 主体
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])

        # 输出层
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.lr_head = nn.Linear(args.n_embd, self.lr_dim)
        self.hr_head = nn.Linear(args.n_embd, self.hr_dim)

        # 平滑卷积（仅用于 HR）
        self.smooth = nn.Sequential(
            nn.Conv2d(4, 4, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(4, 4, 3, padding=1, bias=False),
        )

        from torchmetrics.image import (
            PeakSignalNoiseRatio,
            StructuralSimilarityIndexMeasure,
        )
        from torchmetrics import MeanSquaredError

        self.test_psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.test_ssim = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.test_rmse = MeanSquaredError(squared=False)

    def forward(self, x):
        """因果序列前向"""
        v_first = torch.empty_like(x)
        for block in self.blocks:
            if self.args.grad_cp == 1:
                x, v_first = deepspeed.checkpointing.checkpoint(block, x, v_first)
            else:
                x, v_first = block(x, v_first)
        x = self.ln_out(x)
        return x

    def tokenize_image(self, img, is_hr):
        """将单张影像转换为 token 序列 (B, N, d_model)"""
        if is_hr:
            patch_size = self.patch_size_hr
            proj = self.hr_proj
        else:
            patch_size = self.patch_size_lr
            proj = self.lr_proj
        patches = F.unfold(img, kernel_size=patch_size, stride=patch_size)
        patches = patches.transpose(1, 2)  # (B, N, dim)
        return proj(patches)

    def detokenize_image(self, tokens, is_hr):
        """将 token 序列还原为图像"""
        if is_hr:
            head = self.hr_head
            out_dim = self.hr_dim
            patch_size = self.patch_size_hr
            output_size = self.hr_size
            channels = 4
        else:
            head = self.lr_head
            out_dim = self.lr_dim
            patch_size = self.patch_size_lr
            output_size = self.lr_size
            channels = 8

        patches = head(tokens)
        patches = patches.transpose(1, 2)
        img = F.fold(
            patches,
            output_size=(output_size, output_size),
            kernel_size=patch_size,
            stride=patch_size,
        )
        if is_hr:
            img = self.smooth(img)
        return img

    def upsample_lr_tokens(self, lr_tokens):
        B, N, D = lr_tokens.shape
        H_lr = W_lr = self.lr_grid_size
        H_hr = W_hr = self.hr_grid_size

        # 安全检查：token 数量必须与空间网格匹配
        assert N == H_lr * W_lr, f"lr_tokens has {N} tokens, expected {H_lr*W_lr}"

        # 折叠为特征图: (B, D, H_lr, W_lr)
        feat = lr_tokens.reshape(B, H_lr, W_lr, D).permute(0, 3, 1, 2)
        # 双线性插值到 HR 空间尺寸
        up = F.interpolate(
            feat, size=(H_hr, W_hr), mode="bilinear", align_corners=False
        )
        # 展开回 token 序列: (B, hr_patches, D)
        return up.permute(0, 2, 3, 1).reshape(B, -1, D)

    def training_step(self, batch, batch_idx):
        lr1, hr1, lr2, hr2 = batch
        B = lr1.size(0)

        # ----- 1. 随机选择一个旋转角度 (0, 90, 180, 270) -----
        k = torch.randint(0, 4, (1,)).item()  # 随机整数 0~3
        # -------------------------------------------------

        # 旋转图像
        lr1_r = torch.rot90(lr1, k, [2, 3])
        hr1_r = torch.rot90(hr1, k, [2, 3])
        lr2_r = torch.rot90(lr2, k, [2, 3])
        hr2_r = torch.rot90(hr2, k, [2, 3])

        # ----- 2. Tokenize -----
        lr1_tok = self.tokenize_image(lr1_r, False)
        hr1_tok = self.tokenize_image(hr1_r, True)
        lr2_tok = self.tokenize_image(lr2_r, False)
        hr2_tok = self.tokenize_image(hr2_r, True)

        # ----- 3. Scheduled Sampling -----
        hr2_base = self.upsample_lr_tokens(lr2_tok)
        if torch.rand(1).item() < self.args.ss_prob:
            hr2_inp = hr2_tok
        else:
            hr2_inp = hr2_base

        # ----- 4. 拼接序列并前向 -----
        full_tokens = torch.cat([lr1_tok, hr1_tok, lr2_tok, hr2_inp], dim=1)
        out = self(full_tokens)

        # ----- 5. 提取 hr2 预测部分 -----
        pred = out[:, : self.total_seq_len - 1, :]
        pred_hr2_start = self.hr2_start_idx - 1
        pred_hr2_tok = pred[:, pred_hr2_start : pred_hr2_start + self.hr_patches, :]

        # ----- 6. 还原为图像（仅当前角度）-----
        pred_hr2_r = self.detokenize_image(pred_hr2_tok, is_hr=True)

        # ----- 7. 反向旋转回原角度 -----
        back_k = (-k) % 4
        pred_hr2_orig = torch.rot90(pred_hr2_r, back_k, [2, 3])

        # ----- 8. 损失（直接与原始 hr2 比较）-----
        loss = F.l1_loss(pred_hr2_orig, hr2)
        self.log("train_loss", loss, prog_bar=True, on_step=True)
        return loss

    def generate_hr2(self, lr1, hr1, lr2):
        B = lr1.shape[0]

        # Tokenize
        lr1_tok = self.tokenize_image(lr1, False)
        hr1_tok = self.tokenize_image(hr1, True)
        lr2_tok = self.tokenize_image(lr2, False)

        # 占位符：上采样后的 lr2
        placeholder = self.upsample_lr_tokens(lr2_tok)  # (B, hr_patches, C)

        # 拼接完整序列
        full_seq = torch.cat([lr1_tok, hr1_tok, lr2_tok, placeholder], dim=1)

        # 一次性前向
        out = self(full_seq)

        # 提取 hr2 部分的预测（注意：输出是 shifted 一位的，取 pred 对应位置）
        pred = out[
            :, : self.total_seq_len - 1, :
        ]  # 去掉最后一个预测（它对应 placeholder 后的空）
        pred_hr2_start = self.hr2_start_idx - 1  # 与训练完全一致
        hr2_tokens = pred[:, pred_hr2_start : pred_hr2_start + self.hr_patches, :]

        hr2 = self.detokenize_image(hr2_tokens, is_hr=True)
        return torch.clamp(hr2, 0.0, 1.0)

    def validation_step(self, batch, batch_idx):
        lr1, hr1, lr2, hr2 = batch
        sr_hr2 = self.generate_hr2(lr1, hr1, lr2)
        loss = F.l1_loss(sr_hr2, hr2)
        self.log("val_loss", loss, sync_dist=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        lr1, hr1, lr2, hr2 = batch
        sr_hr2 = self.generate_hr2(lr1, hr1, lr2)
        loss = F.l1_loss(sr_hr2, hr2)
        self.log("test_loss", loss, on_step=True, on_epoch=True)
        sr_hr2 = torch.clamp(sr_hr2, 0.0, 1.0)
        hr2 = torch.clamp(hr2, 0.0, 1.0)
        self.test_psnr.update(sr_hr2, hr2)
        self.test_ssim.update(sr_hr2, hr2)
        self.test_rmse.update(sr_hr2, hr2)
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
        print(
            f"\nTest Results - PSNR: {avg_psnr:.4f} dB, SSIM: {avg_ssim:.4f}, RMSE: {avg_rmse:.6f}"
        )

    def configure_optimizers(self):
        args = self.args
        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        for n, p in self.named_parameters():
            if "att.w0" in n:
                lr_2x.add(n)
            elif (
                (len(p.squeeze().shape) >= 2)
                and (args.weight_decay > 0)
                and ".weight" in n
            ):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))

        param_dict = {n: p for n, p in self.named_parameters()}

        optim_groups = [
            {
                "params": [param_dict[n] for n in lr_1x],
                "lr": args.lr_init * 1.0,
                "weight_decay": 0.0,
            },
            {
                "params": [param_dict[n] for n in lr_2x],
                "lr": args.lr_init * 2.0,
                "weight_decay": 0.0,
            },
        ]
        if args.weight_decay > 0:
            optim_groups.append(
                {
                    "params": [param_dict[n] for n in lr_decay],
                    "lr": args.lr_init * 1.0,
                    "weight_decay": args.weight_decay,
                }
            )

        # 选择优化器
        if self.deepspeed_offload:
            optimizer = DeepSpeedCPUAdam(
                optim_groups,
                betas=args.betas,
                eps=args.adam_eps,
                bias_correction=True,
                adamw_mode=True,
                amsgrad=False,
            )
        else:
            optimizer = FusedAdam(
                optim_groups,
                betas=args.betas,
                eps=args.adam_eps,
                bias_correction=True,
                adam_w_mode=True,
                amsgrad=False,
            )
        from torch.optim.lr_scheduler import ReduceLROnPlateau

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",  # 因为要最小化 val_loss
            factor=0.75,  # 触发时学习率乘以 0.75
            patience=1,  # val_loss 连续 1 个 epoch 不下降才降低
            threshold=1e-4,  # 最小改善量，低于此视为未改善
            min_lr=1e-6,  # 学习率下限
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",  # 监控验证损失
                "interval": "epoch",  # 每个 epoch 检查一次
                "frequency": 1,  # 每1个epoch检查
            },
        }

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False
