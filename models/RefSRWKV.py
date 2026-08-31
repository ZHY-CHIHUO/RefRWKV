# Copyright (c) Shanghai AI Lab. All rights reserved.
"""
RefSRWKV: Reference-based Super-Resolution with RWKV Backbone.
"""
import torch
torch.set_float32_matmul_precision("high")
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import pytorch_lightning as pl
from torch.utils.cpp_extension import load
import os
import math

_cuda_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuda")

# ═══════════════════════════════════════════════════════════
# CUDA WKV 算子封装
# ═══════════════════════════════════════════════════════════
_wkv_cuda = None
_wkv_load_error = None

def _get_wkv_cuda():
    global _wkv_cuda, _wkv_load_error
    if _wkv_cuda is not None:
        return _wkv_cuda
    if _wkv_load_error is not None:
        raise RuntimeError("Bi-WKV CUDA 扩展加载失败") from _wkv_load_error
    if not torch.cuda.is_available():
        raise RuntimeError("Bi-WKV 需要 CUDA 环境")
    cap = torch.cuda.get_device_capability()
    arch = f"compute_{cap[0]}{cap[1]}"
    sm = f"sm_{cap[0]}{cap[1]}"
    _wkv_cuda = load(
        name="bi_wkv",
        sources=[
            os.path.join(_cuda_dir, "bi_wkv.cpp"),
            os.path.join(_cuda_dir, "bi_wkv_kernel.cu"),
        ],
        verbose=True,
        extra_cuda_cflags=[
            "-res-usage", "--maxrregcount 60", "--use_fast_math", "-O3", "-Xptxas -O3",
            f"-gencode arch={arch},code={sm}", f"-gencode arch={arch},code={arch}",
        ],
    )
    return _wkv_cuda

try:
    _compiler_disable = torch.compiler.disable
except AttributeError:
    def _compiler_disable(fn=None, **kwargs):
        return fn if fn is not None else (lambda f: f)

class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, u, k, v):
        half_mode = w.dtype == torch.half
        bf_mode = w.dtype == torch.bfloat16
        ctx.save_for_backward(w, u, k, v)
        y = _get_wkv_cuda().bi_wkv_forward(
            w.float().contiguous(), u.float().contiguous(),
            k.float().contiguous(), v.float().contiguous()
        )
        if half_mode: return y.half()
        if bf_mode: return y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        w, u, k, v = ctx.saved_tensors
        half_mode = w.dtype == torch.half
        bf_mode = w.dtype == torch.bfloat16
        gw, gu, gk, gv = _get_wkv_cuda().bi_wkv_backward(
            w.float().contiguous(), u.float().contiguous(),
            k.float().contiguous(), v.float().contiguous(),
            gy.float().contiguous(),
        )
        if half_mode: return (gw.half(), gu.half(), gk.half(), gv.half())
        if bf_mode: return (gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        return (gw, gu, gk, gv)

@_compiler_disable()
def RUN_CUDA(w, u, k, v):
    return WKV.apply(w.float(), u.float(), k.float(), v.float())

# ═══════════════════════════════════════════════════════════
# 基础组件
# ═══════════════════════════════════════════════════════════
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor

class OmniShift(nn.Module):
    """多尺度深度卷积融合，推理时重参数化为单个 5×5 卷积以提升速度。"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.conv1x1 = nn.Conv2d(dim, dim, 1, groups=dim, bias=False)
        self.conv3x3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        self.conv5x5 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim, bias=False)
        self.alpha = nn.Parameter(torch.ones(4) * 0.25)
        self.gate = nn.Parameter(torch.zeros(1))
        self.register_buffer("conv5x5_reparam_weight", torch.zeros(dim, 1, 5, 5))
        self._reparam_done = False

    def forward_train(self, x):
        alpha = torch.softmax(self.alpha, dim=0)
        shifted = (
            alpha[0] * x + alpha[1] * self.conv1x1(x) +
            alpha[2] * self.conv3x3(x) + alpha[3] * self.conv5x5(x)
        )
        return x + torch.tanh(self.gate) * (shifted - x)

    def reparam_5x5(self):
        if self._reparam_done:
            return
        with torch.no_grad():
            g = torch.tanh(self.gate)
            alpha = torch.softmax(self.alpha, dim=0)
            identity = torch.zeros(self.dim, 1, 5, 5, device=self.conv1x1.weight.device)
            identity[:, :, 2, 2] = 1.0
            w1 = F.pad(self.conv1x1.weight, (2, 2, 2, 2))
            w3 = F.pad(self.conv3x3.weight, (1, 1, 1, 1))
            w5 = self.conv5x5.weight
            shifted = alpha[0]*identity + alpha[1]*w1 + alpha[2]*w3 + alpha[3]*w5
            combined = (1.0 - g)*identity + g*shifted
            self.conv5x5_reparam_weight.copy_(combined)
        self._reparam_done = True

    def forward(self, x):
        if self.training:
            self._reparam_done = False
            return self.forward_train(x)
        if not self._reparam_done:
            self.reparam_5x5()
        return F.conv2d(x, self.conv5x5_reparam_weight, padding=2, groups=self.dim)

# ═══════════════════════════════════════════════════════════
# RWKV 空间与通道混合模块
# ═══════════════════════════════════════════════════════════
class VRWKV_SpatialMix(nn.Module):
    """窗口化 RWKV 空间混合，采用 8×8 窗口与循环移位机制。"""
    def __init__(self, n_embd, head_dim=64, window_size=8, shift_size=3, num_groups=None):
        super().__init__()
        if num_groups is None:
            num_groups = max(1, n_embd // 16)
        assert n_embd % num_groups == 0, f"n_embd({n_embd}) 必须被 num_groups({num_groups}) 整除"
        assert n_embd % 16 == 0, f"n_embd({n_embd}) 必须是 16 的倍数以适配 CUDA kernel"

        self.n_embd = n_embd
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_groups = num_groups
        self.group_dim = n_embd // num_groups
        self.recurrence = 2

        self.omni_shift = OmniShift(dim=n_embd)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.output = nn.Linear(n_embd, n_embd, bias=False)
        self.register_buffer("scale", torch.tensor(n_embd ** 0.5))

        # 通道分段初始化，赋予不同组不同的衰减初值
        with torch.no_grad():
            decay_init = torch.zeros(self.recurrence, n_embd)
            for g in range(num_groups):
                s = g * self.group_dim
                decay_init[:, s:s + self.group_dim] = -0.5 * (g + 1)
            self.spatial_decay = nn.Parameter(decay_init)
            self.spatial_first = nn.Parameter(torch.zeros(self.recurrence, n_embd))

        mid_ch = max(n_embd // 4, 8)
        self.channel_gate = nn.Sequential(
            nn.Conv2d(n_embd, mid_ch, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, n_embd, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def jit_func(self, x, resolution):
        h, w = resolution
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.omni_shift(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        return self.key(x), self.value(x), torch.sigmoid(self.receptance(x))

    def _window_wkv(self, k, v, sr):
        s = self.scale
        ws = self.window_size
        for j in range(self.recurrence):
            dj = self.spatial_decay[j] / s
            fj = self.spatial_first[j] / s
            if j % 2 == 0:
                v = RUN_CUDA(dj, fj, k, v)
            else:
                kt = rearrange(k, "b (h w) c -> b (w h) c", h=ws, w=ws)
                vt = rearrange(v, "b (h w) c -> b (w h) c", h=ws, w=ws)
                v = rearrange(RUN_CUDA(dj, fj, kt, vt), "b (w h) c -> b (h w) c", h=ws, w=ws)
        return sr * v

    def forward(self, x, resolution, layer_idx=0):
        B, T, C = x.size()
        h, w = resolution
        ws = self.window_size
        ss = self.shift_size

        sr, k, v = self.jit_func(x, resolution)
        shift_amt = (layer_idx % 3) * ss

        k = rearrange(k, "b (hh ww) c -> b hh ww c", hh=h, ww=w)
        v = rearrange(v, "b (hh ww) c -> b hh ww c", hh=h, ww=w)
        sr = rearrange(sr, "b (hh ww) c -> b hh ww c", hh=h, ww=w)

        # 智能 Padding 以适配窗口划分与移位
        target_h = max(h, h + shift_amt)
        target_w = max(w, w + shift_amt)
        pad_h = (ws - target_h % ws) % ws
        pad_w = (ws - target_w % ws) % ws

        if pad_h > 0 or pad_w > 0:
            k = F.pad(k, (0, 0, 0, pad_w, 0, pad_h))
            v = F.pad(v, (0, 0, 0, pad_w, 0, pad_h))
            sr = F.pad(sr, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = h + pad_h, w + pad_w

        k = rearrange(k, "b (nh w1) (nw w2) c -> (b nh nw) (w1 w2) c", w1=ws, w2=ws)
        v = rearrange(v, "b (nh w1) (nw w2) c -> (b nh nw) (w1 w2) c", w1=ws, w2=ws)
        sr = rearrange(sr, "b (nh w1) (nw w2) c -> (b nh nw) (w1 w2) c", w1=ws, w2=ws)

        out = self._window_wkv(k, v, sr)

        out = rearrange(out, "(b nh nw) (w1 w2) c -> b (nh w1) (nw w2) c",
                        nh=Hp // ws, nw=Wp // ws, w1=ws, w2=ws)

        if pad_h > 0 or pad_w > 0:
            out = out[:, :h, :w, :]

        gate_in = out.permute(0, 3, 1, 2)
        gate = self.channel_gate(gate_in)
        out = out * gate.permute(0, 2, 3, 1)

        out = rearrange(out, "b hh ww c -> b (hh ww) c")
        return self.output(out)

class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, hidden_rate=4):
        super().__init__()
        self.n_embd = n_embd
        hidden_sz = int(hidden_rate * n_embd)
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)
        self.omni_shift = OmniShift(dim=n_embd)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

    def forward(self, x, resolution):
        h, w = resolution
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.omni_shift(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        k = self.key(x)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        x = torch.sigmoid(self.receptance(x)) * kv
        return x

class Block(nn.Module):
    def __init__(self, n_embd, hidden_rate=4, drop_path=0.0, layer_idx=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.att = VRWKV_SpatialMix(n_embd)
        self.ffn = VRWKV_ChannelMix(n_embd, hidden_rate)
        self.gamma1 = nn.Parameter(torch.ones(n_embd), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones(n_embd), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        b, c, h, w = x.shape
        resolution = (h, w)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = x + self.drop_path(self.gamma1 * self.att(self.ln1(x), resolution, self.layer_idx))
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = x + self.drop_path(self.gamma2 * self.ffn(self.ln2(x), resolution))
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        return x

# ═══════════════════════════════════════════════════════════
# 空间缩放与特征融合
# ═══════════════════════════════════════════════════════════
class Downsample(nn.Module):
    def __init__(self, n_feat, channel_scale=2):
        super().__init__()
        mid_channels = n_feat * channel_scale // 4
        assert mid_channels > 0
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, mid_channels, 3, 1, 1, bias=False),
            nn.PixelUnshuffle(2),
        )
    def forward(self, x): return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat, channel_scale=0.5):
        super().__init__()
        mid_channels = int(n_feat * channel_scale * 4)
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, mid_channels, 3, 1, 1, bias=False),
            nn.PixelShuffle(2),
        )
    def forward(self, x): return self.body(x)

def _gn_groups(num_channels: int, max_groups: int = 32) -> int:
    for g in range(min(max_groups, num_channels), 0, -1):
        if num_channels % g == 0: return g
    return 1

class GatedFusion(nn.Module):
    """基于余弦相似度的置信度门控融合模块。"""
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.fuse_conv = nn.Conv2d(dim * 2, dim, 1, bias=False)
        self.norm = nn.GroupNorm(_gn_groups(dim), dim)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, max(dim // reduction, 8), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(dim // reduction, 8), dim, 1),
            nn.Sigmoid(),
        )
        nn.init.trunc_normal_(self.fuse_conv.weight, std=0.02)
        nn.init.constant_(self.gate[-2].bias, 0.0)

    def forward(self, lr_feat, ref_feat):
        sim = F.cosine_similarity(lr_feat, ref_feat, dim=1).unsqueeze(1)
        conf = torch.sigmoid(sim * 2.0)
        fused = self.fuse_conv(torch.cat([lr_feat, ref_feat], dim=1))
        fused = self.norm(fused)
        gate = self.gate(fused) * conf
        return lr_feat + gate * fused

# ═══════════════════════════════════════════════════════════
# 核心超分网络
# ═══════════════════════════════════════════════════════════
class RefSRWKV(nn.Module):
    def __init__(
        self,
        inp_channels: int = 3,
        out_channels: int = 3,
        dim: int = 48,
        num_blocks: tuple = (4, 6, 6, 8),
        num_refinement_blocks: int = 4,
        scale: int = 4,
        hr_size: int = 480,
        drop_path_rate: float = 0.1,
        hidden_rate: int = 4,
    ):
        super().__init__()
        self.scale = scale
        self.dim = dim
        self.out_channels = out_channels
        
        # ★ 核心自动化逻辑：绑定 PixelUnshuffle(4) 黄金法则
        self.ref_down_factor = 4  
        assert hr_size % self.ref_down_factor == 0, \
            f"HR尺寸({hr_size}) 必须能被 {self.ref_down_factor} 整除"
        self.internal_size = hr_size // self.ref_down_factor

        # LR 特征提取
        self.lr_up = nn.Sequential(
            nn.Conv2d(inp_channels, dim, 3, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(dim), dim), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(dim), dim), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(_gn_groups(dim), dim), nn.ReLU(inplace=True),
        )

        # Ref 特征提取（利用 PixelUnshuffle 进行无损空间折叠）
        self.ref_to_level1 = nn.Sequential(
            nn.PixelUnshuffle(self.ref_down_factor),
            nn.Conv2d(out_channels * (self.ref_down_factor ** 2), dim, 1, bias=False),
            nn.GroupNorm(_gn_groups(dim), dim), nn.ReLU(inplace=True),
        )
        self.ref_down2 = nn.Sequential(nn.Conv2d(dim, dim * 2, 3, stride=2, padding=1, bias=False), nn.GroupNorm(_gn_groups(dim * 2), dim * 2))
        self.ref_down3 = nn.Sequential(nn.Conv2d(dim * 2, dim * 4, 3, stride=2, padding=1, bias=False), nn.GroupNorm(_gn_groups(dim * 4), dim * 4))
        self.ref_down4 = nn.Sequential(nn.Conv2d(dim * 4, dim * 8, 3, stride=2, padding=1, bias=False), nn.GroupNorm(_gn_groups(dim * 8), dim * 8))

        self.fuse1 = GatedFusion(dim)
        self.fuse2 = GatedFusion(dim * 2)
        self.fuse3 = GatedFusion(dim * 4)
        self.fuse4 = GatedFusion(dim * 8)

        dp_rates = [drop_path_rate * i / (sum(num_blocks) - 1) for i in range(sum(num_blocks))]
        dp_idx = 0
        global_layer_idx = 0

        # 编码器
        self.encoder_level1 = nn.Sequential(*[Block(dim, hidden_rate, dp_rates[dp_idx + i], layer_idx=global_layer_idx + i) for i in range(num_blocks[0])])
        global_layer_idx += num_blocks[0]; dp_idx += num_blocks[0]
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[Block(dim * 2, hidden_rate, dp_rates[dp_idx + i], layer_idx=global_layer_idx + i) for i in range(num_blocks[1])])
        global_layer_idx += num_blocks[1]; dp_idx += num_blocks[1]
        self.down2_3 = Downsample(dim * 2)
        self.encoder_level3 = nn.Sequential(*[Block(dim * 4, hidden_rate, dp_rates[dp_idx + i], layer_idx=global_layer_idx + i) for i in range(num_blocks[2])])
        global_layer_idx += num_blocks[2]; dp_idx += num_blocks[2]
        self.down3_4 = Downsample(dim * 4)
        self.latent = nn.Sequential(*[Block(dim * 8, hidden_rate, dp_rates[dp_idx + i], layer_idx=global_layer_idx + i) for i in range(num_blocks[3])])
        global_layer_idx += num_blocks[3]; dp_idx += num_blocks[3]

        # 解码器
        self.up4_3 = Upsample(dim * 8)
        self.reduce_chan_level3 = nn.Sequential(nn.Conv2d(dim * 8, dim * 4, 1, bias=False), nn.GroupNorm(_gn_groups(dim * 4), dim * 4))
        self.decoder_level3 = nn.Sequential(*[Block(dim * 4, hidden_rate, drop_path=0.0, layer_idx=global_layer_idx + i) for i in range(num_blocks[2])])
        global_layer_idx += num_blocks[2]
        self.up3_2 = Upsample(dim * 4)
        self.reduce_chan_level2 = nn.Sequential(nn.Conv2d(dim * 4, dim * 2, 1, bias=False), nn.GroupNorm(_gn_groups(dim * 2), dim * 2))
        self.decoder_level2 = nn.Sequential(*[Block(dim * 2, hidden_rate, drop_path=0.0, layer_idx=global_layer_idx + i) for i in range(num_blocks[1])])
        global_layer_idx += num_blocks[1]
        self.up2_1 = Upsample(dim * 2)
        self.reduce_chan_level1 = nn.Sequential(nn.Conv2d(dim * 2, dim, 1, bias=False), nn.GroupNorm(_gn_groups(dim), dim))
        self.decoder_level1 = nn.Sequential(*[Block(dim, hidden_rate, drop_path=0.0, layer_idx=global_layer_idx + i) for i in range(num_blocks[0])])
        global_layer_idx += num_blocks[0]

        self.refinement = nn.Sequential(*[Block(dim, hidden_rate, drop_path=0.0, layer_idx=global_layer_idx + i) for i in range(num_refinement_blocks)])
        
        self.output_head = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, bias=False),
            nn.GroupNorm(_gn_groups(dim), dim), nn.ReLU(inplace=True),
            nn.Conv2d(dim, out_channels, 3, padding=1, bias=False),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            nn.init.zeros_(m.bias); nn.init.ones_(m.weight)

    def _match_color(self, ref, target):
        ref_mean = ref.mean(dim=[2, 3], keepdim=True)
        ref_std = ref.std(dim=[2, 3], keepdim=True)
        tgt_mean = target.mean(dim=[2, 3], keepdim=True)
        tgt_std = target.std(dim=[2, 3], keepdim=True)
        return (ref - ref_mean) / (ref_std + 1e-6) * tgt_std + tgt_mean

    def _extract_ref_pyramid(self, ref):
        ref_1 = self.ref_to_level1(ref)
        ref_2 = self.ref_down2(ref_1)
        ref_3 = self.ref_down3(ref_2)
        ref_4 = self.ref_down4(ref_3)
        return ref_1, ref_2, ref_3, ref_4

    def forward(self, lr, ref):
        target_hr_h, target_hr_w = ref.shape[2], ref.shape[3]
        lr_hr = F.interpolate(lr, size=(target_hr_h, target_hr_w), mode="bicubic", align_corners=False)
        ref_aligned = self._match_color(ref, lr_hr)

        int_size = self.internal_size
        
        # 统一映射至内部计算尺寸
        if lr.shape[2] != int_size or lr.shape[3] != int_size:
            lr_int = F.interpolate(lr, size=(int_size, int_size), mode="bicubic", align_corners=False)
        else:
            lr_int = lr

        target_ref_size = int_size * self.ref_down_factor
        if ref_aligned.shape[2] != target_ref_size or ref_aligned.shape[3] != target_ref_size:
            ref_int = F.interpolate(ref_aligned, size=(target_ref_size, target_ref_size), mode="bicubic", align_corners=False)
        else:
            ref_int = ref_aligned

        fea = self.lr_up(lr_int)
        ref_1, ref_2, ref_3, ref_4 = self._extract_ref_pyramid(ref_int)

        e1 = self.encoder_level1(self.fuse1(fea, ref_1))
        e2 = self.encoder_level2(self.fuse2(self.down1_2(e1), ref_2))
        e3 = self.encoder_level3(self.fuse3(self.down2_3(e2), ref_3))
        latent = self.latent(self.fuse4(self.down3_4(e3), ref_4))

        d3 = self.decoder_level3(self.reduce_chan_level3(torch.cat([self.up4_3(latent), e3], dim=1)))
        d2 = self.decoder_level2(self.reduce_chan_level2(torch.cat([self.up3_2(d3), e2], dim=1)))
        d1 = self.decoder_level1(self.reduce_chan_level1(torch.cat([self.up2_1(d2), e1], dim=1)))
        d1 = self.refinement(d1)

        out_feat = self.output_head(d1)
        if out_feat.shape[2] != target_hr_h or out_feat.shape[3] != target_hr_w:
            out_feat = F.interpolate(out_feat, size=(target_hr_h, target_hr_w), mode="bicubic", align_corners=False)

        return torch.tanh(lr_hr + out_feat)

    def prepare_for_inference(self):
        self.eval()
        for m in self.modules():
            if isinstance(m, OmniShift): m.reparam_5x5()
        return self

# ═══════════════════════════════════════════════════════════
# EMA 与 Lightning 训练封装
# ═══════════════════════════════════════════════════════════
class EMA:
    def __init__(self, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._initialized = False

    def _lazy_init(self, model: nn.Module):
        if self._initialized: return
        for name, param in model.named_parameters():
            if param.requires_grad: self.shadow[name] = param.data.clone()
        self._initialized = True

    @torch.no_grad()
    def update(self, model: nn.Module):
        self._lazy_init(model)
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                if self.shadow[name].device != param.device:
                    self.shadow[name] = self.shadow[name].to(param.device)
                p_data = param.data.float() if param.data.dtype != torch.float32 else param.data
                self.shadow[name].mul_(self.decay).add_(p_data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                if self.shadow[name].device != param.device:
                    self.shadow[name] = self.shadow[name].to(param.device)
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name].to(param.device))
        self.backup = {}

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow, "initialized": self._initialized}

    def load_state_dict(self, sd):
        self.decay = sd["decay"]
        self.shadow = sd["shadow"]
        self._initialized = sd["initialized"]

class LitRefSRWKV(pl.LightningModule):
    def __init__(
        self, model_sr: RefSRWKV, learning_rate: float = 1e-4, warmup_steps: int = 500,
        grad_clip_norm: float = 1.0, ema_decay: float = 0.999, use_ema: bool = True,
        ssim_weight: float = 0.0, fft_weight: float = 0.0, ref_drop_prob: float = 0.0,
        loss_fn=None, lr_key: str = "lr", hr_key: str = "hr", ref_key: str = "ref",
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model_sr", "loss_fn"])
        self.model_sr = model_sr
        self.ssim_weight = ssim_weight
        self.fft_weight = fft_weight

        if ssim_weight > 0 or fft_weight > 0:
            self.l1_loss = nn.L1Loss()
            if ssim_weight > 0:
                try:
                    from pyiqa import create_metric as _create_pyiqa_metric
                    self.ssim_loss_fn = _create_pyiqa_metric("ssim", loss_mode=True)
                    self._ssim_backend = "pyiqa"
                except Exception:
                    self.ssim_loss_fn = None
                    self._ssim_backend = "manual"
            else:
                self.ssim_loss_fn = None
            self.criterion = None
        else:
            self.criterion = loss_fn or nn.L1Loss()
            self.ssim_loss_fn = None

        self.lr_key = lr_key
        self.hr_key = hr_key
        self.ref_key = ref_key
        self.ema = EMA(decay=ema_decay) if use_ema else None

    def _unpack_batch(self, batch):
        if isinstance(batch, dict):
            return batch[self.lr_key], batch[self.hr_key], batch[self.ref_key]
        return batch[0], batch[1], batch[2]

    def _apply_ref_dropout(self, ref):
        p = self.hparams.ref_drop_prob
        if p <= 0 or not self.training or ref.size(0) < 2:
            return ref
        drop = (torch.rand(ref.size(0), 1, 1, 1, device=ref.device) < p).float()
        shuffled = ref[torch.randperm(ref.size(0), device=ref.device)]
        return drop * shuffled + (1.0 - drop) * ref

    def forward(self, lr, ref):
        return self.model_sr(lr, ref)

    @staticmethod
    def _fft_loss(pred, target):
        pred_fft = torch.fft.rfft2(pred, norm="ortho")
        target_fft = torch.fft.rfft2(target, norm="ortho")
        return F.l1_loss(pred_fft, target_fft)

    def training_step(self, batch, batch_idx):
        lr, hr, ref = self._unpack_batch(batch)
        ref = self._apply_ref_dropout(ref)
        output = self(lr, ref)

        if self.ssim_weight > 0 or self.fft_weight > 0:
            l1_loss = self.l1_loss(output, hr)
            loss = l1_loss
            if self.ssim_weight > 0:
                if self.ssim_loss_fn is not None and self._ssim_backend == "pyiqa":
                    ssim_loss = 1.0 - self.ssim_loss_fn(output, hr)
                else:
                    ssim_loss = self._manual_ssim_loss(output, hr)
                loss = loss + self.ssim_weight * ssim_loss
                self.log("train_ssim_loss", ssim_loss, on_step=True, on_epoch=True)
            if self.fft_weight > 0:
                fft_loss = self._fft_loss(output, hr)
                loss = loss + self.fft_weight * fft_loss
                self.log("train_fft_loss", fft_loss, on_step=True, on_epoch=True)
            self.log("train_l1", l1_loss, on_step=True, on_epoch=True)
        else:
            loss = self.criterion(output, hr)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    @staticmethod
    def _manual_ssim_loss(pred, target):
        C = pred.shape[1]
        window_size = 11
        sigma = 1.5
        coords = torch.arange(window_size, dtype=pred.dtype, device=pred.device)
        g = torch.exp(-((coords - window_size // 2) ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.unsqueeze(0) * g.unsqueeze(1)
        window = window.unsqueeze(0).unsqueeze(0).repeat(C, 1, 1, 1)
        pad = window_size // 2
        mu_pred = F.conv2d(pred, window, padding=pad, groups=C)
        mu_target = F.conv2d(target, window, padding=pad, groups=C)
        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target
        sigma_pred_sq = F.conv2d(pred ** 2, window, padding=pad, groups=C) - mu_pred_sq
        sigma_target_sq = F.conv2d(target ** 2, window, padding=pad, groups=C) - mu_target_sq
        sigma_pred_target = F.conv2d(pred * target, window, padding=pad, groups=C) - mu_pred_target
        C1 = (0.01 * 2.0) ** 2
        C2 = (0.03 * 2.0) ** 2
        ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / (
            (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
        )
        return 1.0 - ssim_map.mean()

    def on_train_batch_start(self, batch, batch_idx):
        if self.global_step < self.hparams.warmup_steps:
            progress = (self.global_step + 1) / self.hparams.warmup_steps
            lr_scale = 1e-3 + (1.0 - 1e-3) * progress
            for pg in self.optimizers().param_groups:
                pg["lr"] = self.hparams.learning_rate * lr_scale

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema is not None:
            self.ema.update(self.model_sr)

    def on_validation_epoch_start(self):
        if self.ema is not None:
            self.ema.apply_shadow(self.model_sr)

    def validation_step(self, batch, batch_idx):
        lr, hr, ref = self._unpack_batch(batch)
        output = self(lr, ref)
        if self.ssim_weight > 0 or self.fft_weight > 0:
            loss = self.l1_loss(output, hr)
            self.log("val_l1", loss, on_step=False, on_epoch=True)
            if self.ssim_weight > 0:
                if self.ssim_loss_fn is not None and self._ssim_backend == "pyiqa":
                    ssim_loss = 1.0 - self.ssim_loss_fn(output, hr)
                else:
                    ssim_loss = self._manual_ssim_loss(output, hr)
                loss = loss + self.ssim_weight * ssim_loss
                self.log("val_ssim_loss", ssim_loss, on_step=False, on_epoch=True)
            if self.fft_weight > 0:
                fft_loss = self._fft_loss(output, hr)
                loss = loss + self.fft_weight * fft_loss
                self.log("val_fft_loss", fft_loss, on_step=False, on_epoch=True)
        else:
            loss = self.criterion(output, hr)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        mse = F.mse_loss(output, hr)
        psnr = 10 * torch.log10(4.0 / (mse + 1e-8))
        self.log("val/psnr", psnr, on_step=False, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        if self.ema is not None:
            self.ema.restore(self.model_sr)

    def on_test_epoch_start(self):
        if self.ema is not None:
            self.ema.apply_shadow(self.model_sr)

    def test_step(self, batch, batch_idx):
        lr, hr, ref = self._unpack_batch(batch)
        output = self(lr, ref)
        loss = (self.criterion or self.l1_loss)(output, hr)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return output, hr

    def on_test_epoch_end(self):
        if self.ema is not None:
            self.ema.restore(self.model_sr)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=(
                    self.trainer.estimated_stepping_batches
                    if hasattr(self.trainer, "estimated_stepping_batches")
                    else 100000
                ),
                eta_min=1e-6,
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None
    ):
        clip_val = gradient_clip_val or self.hparams.grad_clip_norm
        if clip_val and clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=clip_val,
                gradient_clip_algorithm=gradient_clip_algorithm or "norm",
            )

    def on_save_checkpoint(self, checkpoint):
        if self.ema is not None:
            checkpoint["ema_state_dict"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if self.ema is not None and "ema_state_dict" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])

    def on_train_start(self):
        total = sum(p.numel() for p in self.parameters())
        ema_info = f" | EMA decay={self.ema.decay}" if self.ema else " | EMA=off"
        ssim_info = f" | SSIM={self.ssim_weight}" if self.ssim_weight > 0 else ""
        fft_info = f" | FFT={self.fft_weight}" if self.fft_weight > 0 else ""
        print(
            f"✅ LitRefSRWKV 训练开始 | 参数量: {total / 1e6:.2f}M"
            f" | grad_clip={self.hparams.grad_clip_norm}{ema_info}{ssim_info}{fft_info}"
        )