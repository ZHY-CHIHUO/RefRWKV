# Copyright (c) Shanghai AI Lab. All rights reserved.
"""
RefSRWKV: Reference-based Super-Resolution with RWKV Backbone.
终极生产版：
  - 自动推导 internal_size (hr_size // 4)，绑定 PixelUnshuffle(4) 黄金法则
  - 8×8 窗口注意力 + 循环移位 + 智能 Padding
  - 通道分段初始化以适配底层 CUDA WKV 算子
  - 修复 GatedFusion 置信度映射，防止参考图信息被过度抑制
  - 修复 EMA 验证期污染与学习率调度冲突
"""
import torch
try:
    torch.set_float32_matmul_precision("high")
except AttributeError:
    pass
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
    if _wkv_cuda is not None: return _wkv_cuda
    if _wkv_load_error is not None: raise RuntimeError("Bi-WKV CUDA 扩展加载失败") from _wkv_load_error
    if not torch.cuda.is_available(): raise RuntimeError("Bi-WKV 需要 CUDA 环境")
    if not os.path.isfile(os.path.join(_cuda_dir, "bi_wkv.cpp")) or not os.path.isfile(os.path.join(_cuda_dir, "bi_wkv_kernel.cu")):
        raise FileNotFoundError(f"Bi-WKV CUDA 源文件不存在: {_cuda_dir}")
    cap = torch.cuda.get_device_capability()
    arch, sm = f"compute_{cap[0]}{cap[1]}", f"sm_{cap[0]}{cap[1]}"
    try:
        _wkv_cuda = load(
            name="bi_wkv",
            sources=[os.path.join(_cuda_dir, "bi_wkv.cpp"), os.path.join(_cuda_dir, "bi_wkv_kernel.cu")],
            verbose=True,
            extra_cuda_cflags=["-res-usage", "--maxrregcount 60", "--use_fast_math", "-O3", "-Xptxas -O3",
                               f"-gencode arch={arch},code={sm}", f"-gencode arch={arch},code={arch}"],
        )
    except Exception as exc:
        _wkv_load_error = exc
        raise RuntimeError(f"Bi-WKV CUDA 扩展编译/加载失败 (sm_{cap[0]}{cap[1]})") from exc
    return _wkv_cuda

try: _compiler_disable = torch.compiler.disable
except AttributeError:
    def _compiler_disable(fn=None, **kwargs): return fn if fn is not None else (lambda f: f)

class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, u, k, v):
        half_mode, bf_mode = w.dtype == torch.half, w.dtype == torch.bfloat16
        ctx.save_for_backward(w, u, k, v)
        y = _get_wkv_cuda().bi_wkv_forward(w.float().contiguous(), u.float().contiguous(), k.float().contiguous(), v.float().contiguous())
        return y.half() if half_mode else (y.bfloat16() if bf_mode else y)
    @staticmethod
    def backward(ctx, gy):
        w, u, k, v = ctx.saved_tensors
        half_mode, bf_mode = w.dtype == torch.half, w.dtype == torch.bfloat16
        gw, gu, gk, gv = _get_wkv_cuda().bi_wkv_backward(w.float().contiguous(), u.float().contiguous(), k.float().contiguous(), v.float().contiguous(), gy.float().contiguous())
        if half_mode: return (gw.half(), gu.half(), gk.half(), gv.half())
        if bf_mode: return (gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        return (gw, gu, gk, gv)

@_compiler_disable()
def RUN_CUDA(w, u, k, v): return WKV.apply(w.float(), u.float(), k.float(), v.float())

# ═══════════════════════════════════════════════════════════
# 基础组件
# ═══════════════════════════════════════════════════════════
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob, self.scale_by_keep = drop_prob, scale_by_keep
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training: return x
        keep_prob = 1.0 - self.drop_prob
        random_tensor = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep: random_tensor.div_(keep_prob)
        return x * random_tensor

class OmniShift(nn.Module):
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
        shifted = alpha[0]*x + alpha[1]*self.conv1x1(x) + alpha[2]*self.conv3x3(x) + alpha[3]*self.conv5x5(x)
        return x + torch.tanh(self.gate) * (shifted - x)
    def reparam_5x5(self):
        if self._reparam_done: return
        with torch.no_grad():
            g, alpha = torch.tanh(self.gate), torch.softmax(self.alpha, dim=0)
            weight = self.conv1x1.weight
            identity = torch.zeros(self.dim, 1, 5, 5, device=weight.device, dtype=weight.dtype)
            identity[:, :, 2, 2] = 1.0
            w1, w3, w5 = F.pad(weight, (2,2,2,2)), F.pad(self.conv3x3.weight, (1,1,1,1)), self.conv5x5.weight
            self.conv5x5_reparam_weight.copy_((1.0-g)*identity + g*(alpha[0]*identity + alpha[1]*w1 + alpha[2]*w3 + alpha[3]*w5))
        self._reparam_done = True
    def forward(self, x):
        if self.training:
            self._reparam_done = False
            return self.forward_train(x)
        if not self._reparam_done: self.reparam_5x5()
        return F.conv2d(x, self.conv5x5_reparam_weight, padding=2, groups=self.dim)

# ═══════════════════════════════════════════════════════════
# RWKV 空间与通道混合模块
# ═══════════════════════════════════════════════════════════
class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd, head_dim=64, window_size=8, shift_size=3, num_groups=None):
        super().__init__()
        if num_groups is None: num_groups = max(1, n_embd // 16)
        if not isinstance(num_groups, int) or num_groups < 1:
            raise ValueError("num_groups 必须为正整数")
        if not isinstance(window_size, int) or not isinstance(shift_size, int):
            raise ValueError("window_size 和 shift_size 必须为整数")
        if n_embd < 16 or n_embd % num_groups != 0 or n_embd % 16 != 0:
            raise ValueError("n_embd 必须被 num_groups 整除，且至少为 16 和 16 的倍数")
        if window_size < 1 or shift_size < 0 or shift_size >= window_size:
            raise ValueError("window_size 必须为正数，且 0 <= shift_size < window_size")
        self.n_embd, self.window_size, self.shift_size = n_embd, window_size, shift_size
        self.num_groups, self.group_dim, self.recurrence = num_groups, n_embd // num_groups, 2
        self.omni_shift = OmniShift(dim=n_embd)
        self.key, self.value, self.receptance, self.output = [nn.Linear(n_embd, n_embd, bias=False) for _ in range(4)]
        self.register_buffer("scale", torch.tensor(n_embd**0.5))
        with torch.no_grad():
            decay_init = torch.zeros(self.recurrence, n_embd)
            # The CUDA kernel expects a positive distance-decay coefficient.
            for g in range(num_groups):
                target_decay = 0.5 * (g + 1)
                decay_init[:, g*self.group_dim:(g+1)*self.group_dim] = math.log(math.expm1(target_decay))
            self.spatial_decay, self.spatial_first = nn.Parameter(decay_init), nn.Parameter(torch.zeros(self.recurrence, n_embd))
        mid_ch = max(n_embd // 4, 8)
        self.channel_gate = nn.Sequential(nn.Conv2d(n_embd, mid_ch, 1, bias=True), nn.ReLU(inplace=True), nn.Conv2d(mid_ch, n_embd, 1, bias=True), nn.Sigmoid())

    def jit_func(self, x, resolution):
        h, w = resolution
        x = self.omni_shift(rearrange(x, "b (h w) c -> b c h w", h=h, w=w))
        x = rearrange(x, "b c h w -> b (h w) c")
        return self.key(x), self.value(x), torch.sigmoid(self.receptance(x))

    def _window_wkv(self, k, v, sr):
        s, ws = self.scale, self.window_size
        for j in range(self.recurrence):
            dj, fj = F.softplus(self.spatial_decay[j]) / s, self.spatial_first[j] / s
            if j % 2 == 0: v = RUN_CUDA(dj, fj, k, v)
            else:
                kt, vt = rearrange(k, "b (h w) c -> b (w h) c", h=ws, w=ws), rearrange(v, "b (h w) c -> b (w h) c", h=ws, w=ws)
                v = rearrange(RUN_CUDA(dj, fj, kt, vt), "b (w h) c -> b (h w) c", h=ws, w=ws)
        return sr * v

    def forward(self, x, resolution, layer_idx=0):
        B, T, C = x.size()
        h, w, ws, ss = resolution[0], resolution[1], self.window_size, self.shift_size
        if T != h * w or C != self.n_embd:
            raise ValueError(f"SpatialMix 输入形状与 resolution 不一致: x={tuple(x.shape)}, resolution={resolution}")
        sr, k, v = self.jit_func(x, resolution)
        shift_amt = (layer_idx % 3) * ss
        k, v, sr = [rearrange(t, "b (hh ww) c -> b hh ww c", hh=h, ww=w) for t in (k, v, sr)]

        # Shift the window origin without circular wrap-around.  Content is
        # padded on the top/left, then the bottom/right is padded to a window
        # multiple; the output is cropped back to the original grid.
        pad_bottom = (ws - (h + shift_amt) % ws) % ws
        pad_right = (ws - (w + shift_amt) % ws) % ws
        if shift_amt or pad_bottom or pad_right:
            pad = (0, 0, shift_amt, pad_right, shift_amt, pad_bottom)
            k, v, sr = [F.pad(t, pad) for t in (k, v, sr)]
        Hp, Wp = h + shift_amt + pad_bottom, w + shift_amt + pad_right
        if Hp % ws or Wp % ws:
            raise RuntimeError(f"窗口 padding 失败: ({Hp}, {Wp}) 不能被窗口 {ws} 整除")

        k, v, sr = [rearrange(t, "b (nh w1) (nw w2) c -> (b nh nw) (w1 w2) c", w1=ws, w2=ws) for t in (k, v, sr)]
        out = self._window_wkv(k, v, sr)
        out = rearrange(out, "(b nh nw) (w1 w2) c -> b (nh w1) (nw w2) c", nh=Hp // ws, nw=Wp // ws, w1=ws, w2=ws)
        out = out[:, shift_amt:shift_amt + h, shift_amt:shift_amt + w, :]

        out = out * self.channel_gate(out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        return self.output(rearrange(out, "b hh ww c -> b (hh ww) c"))

class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, hidden_rate=4):
        super().__init__()
        hidden_sz = int(hidden_rate * n_embd)
        self.key, self.omni_shift, self.receptance, self.value = nn.Linear(n_embd, hidden_sz, bias=False), OmniShift(dim=n_embd), nn.Linear(n_embd, n_embd, bias=False), nn.Linear(hidden_sz, n_embd, bias=False)
    def forward(self, x, resolution):
        h, w = resolution
        x = self.omni_shift(rearrange(x, "b (h w) c -> b c h w", h=h, w=w))
        x = rearrange(x, "b c h w -> b (h w) c")
        return torch.sigmoid(self.receptance(x)) * self.value(torch.square(torch.relu(self.key(x))))

class Block(nn.Module):
    def __init__(self, n_embd, hidden_rate=4, drop_path=0.0, layer_idx=0):
        super().__init__()
        self.layer_idx, self.ln1, self.ln2 = layer_idx, nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.att, self.ffn = VRWKV_SpatialMix(n_embd), VRWKV_ChannelMix(n_embd, hidden_rate)
        self.gamma1, self.gamma2 = nn.Parameter(torch.ones(n_embd)), nn.Parameter(torch.ones(n_embd))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    def forward(self, x):
        b, c, h, w = x.shape
        resolution = (h, w)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = x + self.drop_path(self.gamma1 * self.att(self.ln1(x), resolution, self.layer_idx))
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = x + self.drop_path(self.gamma2 * self.ffn(self.ln2(x), resolution))
        return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

# ═══════════════════════════════════════════════════════════
# 空间缩放与特征融合
# ═══════════════════════════════════════════════════════════
class Downsample(nn.Module):
    def __init__(self, n_feat, channel_scale=2):
        super().__init__()
        mid_channels = n_feat * channel_scale // 4
        self.body = nn.Sequential(nn.Conv2d(n_feat, mid_channels, 3, 1, 1, bias=False), nn.PixelUnshuffle(2))
    def forward(self, x): return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat, channel_scale=0.5):
        super().__init__()
        mid_channels = int(n_feat * channel_scale * 4)
        self.body = nn.Sequential(nn.Conv2d(n_feat, mid_channels, 3, 1, 1, bias=False), nn.PixelShuffle(2))
    def forward(self, x): return self.body(x)

def _gn_groups(num_channels: int, max_groups: int = 32) -> int:
    for g in range(min(max_groups, num_channels), 0, -1):
        if num_channels % g == 0: return g
    return 1

class GatedFusion(nn.Module):
    def __init__(self, dim, reduction=4):
        super().__init__()
        self.fuse_conv = nn.Conv2d(dim * 2, dim, 1, bias=False)
        self.norm = nn.GroupNorm(_gn_groups(dim), dim)
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dim, max(dim // reduction, 8), 1), nn.ReLU(inplace=True), nn.Conv2d(max(dim // reduction, 8), dim, 1), nn.Sigmoid())
        nn.init.trunc_normal_(self.fuse_conv.weight, std=0.02)
        nn.init.constant_(self.gate[-2].bias, 0.0)
    def forward(self, lr_feat, ref_feat):
        sim = F.cosine_similarity(lr_feat, ref_feat, dim=1).unsqueeze(1)
        # ★ 修复：将 [-1, 1] 线性映射到 [0, 1]，避免 sigmoid 导致的过度抑制
        conf = (sim + 1.0) / 2.0  
        fused = self.norm(self.fuse_conv(torch.cat([lr_feat, ref_feat], dim=1)))
        return lr_feat + self.gate(fused) * conf * fused

# ═══════════════════════════════════════════════════════════
# 核心超分网络
# ═══════════════════════════════════════════════════════════
class RefSRWKV(nn.Module):
    def __init__(self, inp_channels: int = 3, out_channels: int = 3, dim: int = 48,
                 num_blocks: tuple = (4, 6, 6, 8), num_refinement_blocks: int = 4,
                 scale: int = 4, hr_size: int = 480,
                 drop_path_rate: float = 0.1, hidden_rate: int = 4,
                 ref_channels: int = None):
        super().__init__()
        if not isinstance(num_blocks, (tuple, list)) or len(num_blocks) != 4:
            raise ValueError("num_blocks 必须包含四个编码器层的 block 数")
        if any(isinstance(n, bool) or not isinstance(n, int) for n in num_blocks):
            raise ValueError("num_blocks 的每一项必须为整数")
        num_blocks = tuple(num_blocks)
        if any(n < 1 for n in num_blocks):
            raise ValueError("num_blocks 的每一项必须为正整数")
        if not isinstance(num_refinement_blocks, int) or num_refinement_blocks < 0:
            raise ValueError("num_refinement_blocks 必须为非负整数")
        if not isinstance(dim, int) or dim < 16 or dim % 16 != 0:
            raise ValueError("dim 必须是至少 16 的 16 倍数，以适配 CUDA WKV")
        if not isinstance(scale, int) or scale < 1:
            raise ValueError("scale 必须为正整数")
        if not isinstance(hr_size, int) or hr_size <= 0:
            raise ValueError("hr_size 必须为正整数")
        if not math.isfinite(float(drop_path_rate)) or not 0.0 <= float(drop_path_rate) < 1.0:
            raise ValueError("drop_path_rate 必须位于 [0, 1)")
        if (not isinstance(hidden_rate, (int, float)) or isinstance(hidden_rate, bool)
                or not math.isfinite(float(hidden_rate)) or hidden_rate <= 0
                or int(hidden_rate * dim) < 1):
            raise ValueError("hidden_rate 必须为正数")
        if ref_channels is None:
            # Reference colour statistics are matched against the LR image,
            # so the reference stream follows the input channel count.
            ref_channels = inp_channels
        if not isinstance(inp_channels, int) or inp_channels < 1:
            raise ValueError("inp_channels 必须为正整数")
        if not isinstance(out_channels, int) or out_channels < 1:
            raise ValueError("out_channels 必须为正整数")
        if not isinstance(ref_channels, int) or ref_channels < 1:
            raise ValueError("ref_channels 必须为正整数")
        if ref_channels != inp_channels:
            raise ValueError("当前颜色对齐路径要求 ref_channels == inp_channels")

        # Three encoder PixelUnshuffle(2) stages and 8x8 WKV windows impose
        # an internal spatial size divisible by 8.  The fixed 4x reference
        # folding then requires HR patches divisible by 32.
        self.ref_down_factor = 4
        if hr_size % (self.ref_down_factor * 8) != 0:
            raise ValueError(
                f"HR尺寸({hr_size}) 必须能被 {self.ref_down_factor * 8} 整除，"
                "否则 U-Net 下采样或 8x8 窗口无法对齐"
            )
        self.scale, self.dim, self.out_channels = scale, dim, out_channels
        self.inp_channels, self.ref_channels = inp_channels, ref_channels
        self.hr_size = hr_size
        self.internal_size = hr_size // self.ref_down_factor
        if inp_channels == out_channels:
            self.skip_proj = nn.Identity()
        else:
            self.skip_proj = nn.Conv2d(inp_channels, out_channels, 1, bias=False)

        self.lr_up = nn.Sequential(
            nn.Conv2d(inp_channels, dim, 3, padding=1, bias=False), nn.GroupNorm(_gn_groups(dim), dim), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False), nn.GroupNorm(_gn_groups(dim), dim), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=2, dilation=2, bias=False), nn.GroupNorm(_gn_groups(dim), dim), nn.ReLU(inplace=True),
        )
        
        self.ref_to_level1 = nn.Sequential(
            nn.PixelUnshuffle(self.ref_down_factor),
            nn.Conv2d(ref_channels * (self.ref_down_factor ** 2), dim, 1, bias=False),
            nn.GroupNorm(_gn_groups(dim), dim), nn.ReLU(inplace=True),
        )
        self.ref_down2 = nn.Sequential(nn.Conv2d(dim, dim * 2, 3, stride=2, padding=1, bias=False), nn.GroupNorm(_gn_groups(dim * 2), dim * 2))
        self.ref_down3 = nn.Sequential(nn.Conv2d(dim * 2, dim * 4, 3, stride=2, padding=1, bias=False), nn.GroupNorm(_gn_groups(dim * 4), dim * 4))
        self.ref_down4 = nn.Sequential(nn.Conv2d(dim * 4, dim * 8, 3, stride=2, padding=1, bias=False), nn.GroupNorm(_gn_groups(dim * 8), dim * 8))
        
        self.fuse1, self.fuse2, self.fuse3, self.fuse4 = GatedFusion(dim), GatedFusion(dim * 2), GatedFusion(dim * 4), GatedFusion(dim * 8)
        
        total_blocks = sum(num_blocks)
        dp_rates = [drop_path_rate * i / max(1, total_blocks - 1) for i in range(total_blocks)]
        dp_idx, global_layer_idx = 0, 0
        
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

        # d1 lives at HR/4.  Learn the missing four spatial phases with two
        # PixelShuffle stages instead of bicubic-resizing a three-channel map.
        # The final residual is zero-initialized so training starts from the
        # actual bicubic skip connection.
        self.up_final = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim * 4, 3, padding=1, bias=False),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
        )
        self.output_conv = nn.Conv2d(dim, out_channels, 3, padding=1, bias=False)
        self.apply(self._init_weights)
        if isinstance(self.skip_proj, nn.Conv2d):
            # Preserve the available channels at initialization; the learned
            # residual then starts from a predictable bicubic baseline.
            nn.init.zeros_(self.skip_proj.weight)
            with torch.no_grad():
                for channel in range(min(inp_channels, out_channels)):
                    self.skip_proj.weight[channel, channel, 0, 0] = 1.0
        nn.init.zeros_(self.output_conv.weight)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)): nn.init.zeros_(m.bias); nn.init.ones_(m.weight)

    def _match_color(self, ref, target):
        # Keep statistics in float32: bf16/half spatial reductions can lose
        # enough precision to create visible colour shifts on flat patches.
        input_dtype = ref.dtype
        ref_f, target_f = ref.float(), target.float()
        ref_mean = ref_f.mean(dim=(2, 3), keepdim=True)
        ref_std = ref_f.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-6)
        tgt_mean = target_f.mean(dim=(2, 3), keepdim=True)
        tgt_std = target_f.std(dim=(2, 3), keepdim=True, unbiased=False).clamp_min(1e-6)
        matched = (ref_f - ref_mean) / ref_std * tgt_std + tgt_mean
        return matched.to(input_dtype)

    def _extract_ref_pyramid(self, ref):
        ref_1 = self.ref_to_level1(ref)
        ref_2 = self.ref_down2(ref_1)
        ref_3 = self.ref_down3(ref_2)
        ref_4 = self.ref_down4(ref_3)
        return ref_1, ref_2, ref_3, ref_4

    def forward(self, lr, ref):
        if lr.ndim != 4 or ref.ndim != 4:
            raise ValueError(f"lr/ref 必须是 4D NCHW 张量，得到 {lr.shape} 和 {ref.shape}")
        if lr.shape[0] != ref.shape[0]:
            raise ValueError(f"lr/ref batch 不一致: {lr.shape[0]} vs {ref.shape[0]}")
        if lr.shape[1] != self.inp_channels:
            raise ValueError(f"lr 通道数应为 {self.inp_channels}，得到 {lr.shape[1]}")
        if ref.shape[1] != self.ref_channels:
            raise ValueError(f"ref 通道数应为 {self.ref_channels}，得到 {ref.shape[1]}")
        if lr.shape[2] < 1 or lr.shape[3] < 1 or ref.shape[2] < 1 or ref.shape[3] < 1:
            raise ValueError("lr/ref 的空间尺寸必须为正数")
        target_hr_h, target_hr_w = ref.shape[2], ref.shape[3]
        lr_hr_input = F.interpolate(lr, size=(target_hr_h, target_hr_w), mode="bicubic", align_corners=False)
        ref_aligned = self._match_color(ref, lr_hr_input)
        lr_hr = self.skip_proj(lr_hr_input)
        
        int_size = self.internal_size
        
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

        out_feat = self.output_conv(self.up_final(d1))
        if out_feat.shape[2] != target_hr_h or out_feat.shape[3] != target_hr_w:
            out_feat = F.interpolate(out_feat, size=(target_hr_h, target_hr_w), mode="bicubic", align_corners=False)

        return torch.clamp(lr_hr + out_feat, min=-1.0, max=1.0)

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
        if not 0.0 <= float(decay) < 1.0:
            raise ValueError("EMA decay 必须位于 [0, 1)")
        self.decay = float(decay)
        self.shadow, self.backup = {}, {}
        self._initialized, self._applied = False, False

    def _lazy_init(self, model: nn.Module):
        # Shadows stay in float32 even when the model is trained with AMP.
        for name, param in model.named_parameters():
            if param.requires_grad and name not in self.shadow:
                self.shadow[name] = param.detach().float().clone()
        self._initialized = True

    @torch.no_grad()
    def update(self, model: nn.Module):
        self._lazy_init(model)
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                if self.shadow[name].device != param.device: self.shadow[name] = self.shadow[name].to(param.device)
                p_data = param.detach().float()
                self.shadow[name].mul_(self.decay).add_(p_data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module):
        if self._applied:
            return
        self._lazy_init(model)
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                if self.shadow[name].device != param.device: self.shadow[name] = self.shadow[name].to(param.device)
                self.backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name].to(device=param.device, dtype=param.dtype))
        self._applied = True

    def restore(self, model: nn.Module):
        if not self._applied:
            return
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup: param.data.copy_(self.backup[name].to(param.device))
        self.backup = {}
        self._applied = False

    def state_dict(self):
        return {
            "decay": self.decay,
            "shadow": {name: value.detach().clone() for name, value in self.shadow.items()},
            "initialized": self._initialized,
        }

    def load_state_dict(self, sd):
        if not isinstance(sd, dict):
            raise TypeError("EMA state 必须是字典")
        self.decay = float(sd.get("decay", self.decay))
        if not 0.0 <= self.decay < 1.0:
            raise ValueError("checkpoint 中的 EMA decay 不在 [0, 1) 范围")
        shadow = sd.get("shadow", {})
        self.shadow = {
            name: value.detach().float().clone()
            for name, value in shadow.items()
            if isinstance(name, str) and torch.is_tensor(value)
        }
        self._initialized = bool(sd.get("initialized", bool(self.shadow)))
        self.backup, self._applied = {}, False

class LitRefSRWKV(pl.LightningModule):
    def __init__(self, model_sr: RefSRWKV, learning_rate: float = 1e-4, warmup_steps: int = 500, grad_clip_norm: float = 1.0, ema_decay: float = 0.999, use_ema: bool = True, ssim_weight: float = 0.0, fft_weight: float = 0.0, ref_drop_prob: float = 0.0, loss_fn=None, lr_key: str = "lr", hr_key: str = "hr", ref_key: str = "ref"):
        super().__init__()
        if not isinstance(model_sr, nn.Module):
            raise TypeError("model_sr 必须是 torch.nn.Module 实例")
        if not float(learning_rate) > 0:
            raise ValueError("learning_rate 必须为正数")
        if not isinstance(warmup_steps, int) or warmup_steps < 0:
            raise ValueError("warmup_steps 必须为非负整数")
        if grad_clip_norm is not None and float(grad_clip_norm) < 0:
            raise ValueError("grad_clip_norm 不能为负数")
        if not 0.0 <= float(ssim_weight) or not 0.0 <= float(fft_weight):
            raise ValueError("ssim_weight 和 fft_weight 不能为负数")
        if not 0.0 <= float(ref_drop_prob) <= 1.0:
            raise ValueError("ref_drop_prob 必须位于 [0, 1]")
        for key in (lr_key, hr_key, ref_key):
            if not isinstance(key, str) or not key:
                raise ValueError("batch key 必须是非空字符串")
        self.save_hyperparameters(ignore=["model_sr", "loss_fn"])
        self.model_sr = model_sr
        self.ssim_weight, self.fft_weight = float(ssim_weight), float(fft_weight)
        self.criterion = loss_fn or nn.L1Loss()
        self.l1_loss = nn.L1Loss()
        self.ssim_loss_fn, self._ssim_backend = None, "manual"
        if self.ssim_weight > 0:
            try:
                from pyiqa import create_metric as _create_pyiqa_metric

                # `as_loss` enables gradients through the metric; it is the
                # supported pyiqa API (the old `loss_mode` argument is not).
                metric = _create_pyiqa_metric("ssim", as_loss=True, device="cpu")
                metric.eval()
                for parameter in metric.parameters():
                    parameter.requires_grad_(False)
                # Keep the optional metric out of Lightning's model
                # state_dict/optimizer.  It is a fixed loss helper, not part
                # of the SR checkpoint and can otherwise make resume checks
                # depend on pyiqa internals.
                object.__setattr__(self, "ssim_loss_fn", metric)
                self._ssim_backend = "pyiqa"
            except Exception:
                self.ssim_loss_fn, self._ssim_backend = None, "manual"
        self.lr_key, self.hr_key, self.ref_key = lr_key, hr_key, ref_key
        self.ema = EMA(decay=float(ema_decay)) if use_ema else None
        self._ema_last_step = -1

    def _unpack_batch(self, batch):
        if isinstance(batch, dict):
            missing = [key for key in (self.lr_key, self.hr_key, self.ref_key) if key not in batch]
            if missing:
                raise KeyError(f"batch 缺少字段: {missing}")
            return batch[self.lr_key], batch[self.hr_key], batch[self.ref_key]
        if not isinstance(batch, (tuple, list)) or len(batch) < 3:
            raise ValueError("batch 必须是包含 lr/hr/ref 的字典或序列")
        return batch[0], batch[1], batch[2]

    def _apply_ref_dropout(self, ref, lr=None):
        p = self.hparams.ref_drop_prob
        if p <= 0 or not self.training:
            return ref
        batch_size = ref.size(0)
        if batch_size >= 2:
            # A one-position roll is a derangement, so a dropped sample can
            # never accidentally receive its own reference image.
            replacement = torch.roll(ref, shifts=1, dims=0)
        elif lr is not None:
            replacement = F.interpolate(lr, size=ref.shape[-2:], mode="bicubic", align_corners=False)
            if replacement.shape[1] != ref.shape[1]:
                replacement = torch.zeros_like(ref)
        else:
            replacement = torch.zeros_like(ref)
        drop = torch.rand(batch_size, 1, 1, 1, device=ref.device) < p
        return torch.where(drop, replacement, ref)

    def forward(self, lr, ref): return self.model_sr(lr, ref)

    @staticmethod
    def _fft_loss(pred, target):
        pred_f, target_f = pred.float(), target.float()
        return (torch.fft.rfft2(pred_f, norm="ortho") - torch.fft.rfft2(target_f, norm="ortho")).abs().mean()

    @staticmethod
    def _manual_ssim_loss(pred, target):
        if pred.ndim != 4 or target.shape != pred.shape:
            raise ValueError(f"SSIM 输入形状不一致: {tuple(pred.shape)} vs {tuple(target.shape)}")
        channels, window_size, sigma = pred.shape[1], 11, 1.5
        pred_f, target_f = pred.float(), target.float()
        coords = torch.arange(window_size, dtype=torch.float32, device=pred.device)
        gaussian = torch.exp(-((coords - window_size // 2) ** 2) / (2.0 * sigma ** 2))
        window_2d = torch.outer(gaussian, gaussian)
        window_2d = window_2d / window_2d.sum()
        window = window_2d.view(1, 1, window_size, window_size).expand(channels, 1, -1, -1).contiguous()
        pad = window_size // 2
        mu_p = F.conv2d(pred_f, window, padding=pad, groups=channels)
        mu_t = F.conv2d(target_f, window, padding=pad, groups=channels)
        sigma_p_sq = (F.conv2d(pred_f.square(), window, padding=pad, groups=channels) - mu_p.square()).clamp_min(0.0)
        sigma_t_sq = (F.conv2d(target_f.square(), window, padding=pad, groups=channels) - mu_t.square()).clamp_min(0.0)
        sigma_pt = F.conv2d(pred_f * target_f, window, padding=pad, groups=channels) - mu_p * mu_t
        c1, c2 = (0.01 * 2.0) ** 2, (0.03 * 2.0) ** 2
        numerator = (2.0 * mu_p * mu_t + c1) * (2.0 * sigma_pt + c2)
        denominator = (mu_p.square() + mu_t.square() + c1) * (sigma_p_sq + sigma_t_sq + c2)
        score = (numerator / denominator.clamp_min(1e-12)).clamp(-1.0, 1.0).mean()
        return 1.0 - score

    def _ssim_loss(self, pred, target):
        if self._ssim_backend == "pyiqa" and self.ssim_loss_fn is not None:
            metric = self.ssim_loss_fn
            metric_device = next(metric.parameters(), None)
            metric_device = metric_device.device if metric_device is not None else pred.device
            if metric_device != pred.device:
                metric.to(pred.device)
            metric.eval()
            pred_01 = ((pred.float() + 1.0) * 0.5).clamp(0.0, 1.0)
            target_01 = ((target.float() + 1.0) * 0.5).clamp(0.0, 1.0)
            score = metric(pred_01, target_01)
            return 1.0 - score.float().mean()
        return self._manual_ssim_loss(pred, target)

    def _compute_loss(self, pred, target):
        if pred.shape != target.shape:
            raise ValueError(f"模型输出与 HR 形状不一致: {tuple(pred.shape)} vs {tuple(target.shape)}")
        base_loss = self.criterion(pred, target)
        if base_loss.ndim:
            base_loss = base_loss.mean()
        total = base_loss
        metrics = {"l1": self.l1_loss(pred, target)}
        if self.ssim_weight > 0:
            metrics["ssim_loss"] = self._ssim_loss(pred, target)
            total = total + self.ssim_weight * metrics["ssim_loss"]
        if self.fft_weight > 0:
            metrics["fft_loss"] = self._fft_loss(pred, target)
            total = total + self.fft_weight * metrics["fft_loss"]
        return total, metrics

    def training_step(self, batch, batch_idx):
        lr, hr, ref = self._unpack_batch(batch)
        output = self(lr, self._apply_ref_dropout(ref, lr=lr))
        loss, metrics = self._compute_loss(output, hr)
        batch_size = hr.size(0)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("train_l1", metrics["l1"], on_step=True, on_epoch=True, batch_size=batch_size)
        if "ssim_loss" in metrics:
            self.log("train_ssim_loss", metrics["ssim_loss"], on_step=True, on_epoch=True, batch_size=batch_size)
        if "fft_loss" in metrics:
            self.log("train_fft_loss", metrics["fft_loss"], on_step=True, on_epoch=True, batch_size=batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        lr, hr, ref = self._unpack_batch(batch)
        output = self(lr, ref)
        loss, metrics = self._compute_loss(output, hr)
        batch_size = hr.size(0)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log("val_l1", metrics["l1"], on_step=False, on_epoch=True, batch_size=batch_size)
        if "ssim_loss" in metrics:
            self.log("val_ssim_loss", metrics["ssim_loss"], on_step=False, on_epoch=True, batch_size=batch_size)
        if "fft_loss" in metrics:
            self.log("val_fft_loss", metrics["fft_loss"], on_step=False, on_epoch=True, batch_size=batch_size)
        mse = F.mse_loss(output.float(), hr.float())
        self.log("val/psnr", 10.0 * torch.log10(4.0 / mse.clamp_min(1e-8)), on_step=False, on_epoch=True, batch_size=batch_size)
        return loss

    def test_step(self, batch, batch_idx):
        lr, hr, ref = self._unpack_batch(batch)
        output = self(lr, ref)
        loss, metrics = self._compute_loss(output, hr)
        batch_size = hr.size(0)
        self.log("test_loss", loss, on_step=False, on_epoch=True, batch_size=batch_size)
        self.log("test_l1", metrics["l1"], on_step=False, on_epoch=True, batch_size=batch_size)
        return loss

    def _ema_apply(self):
        if self.ema:
            self.ema.apply_shadow(self.model_sr)

    def _ema_restore(self):
        if self.ema:
            self.ema.restore(self.model_sr)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Lightning increments global_step only after a real optimizer step.
        # Tracking it avoids updating EMA on accumulated micro-batches and also
        # handles the final step of an epoch without a one-step lag.
        if self.ema:
            current_step = int(self.global_step)
            if current_step > self._ema_last_step:
                self.ema.update(self.model_sr)
                self._ema_last_step = current_step

    # Lightning 2.x model-state hooks, with start/end fallbacks for older
    # releases.  EMA.apply/restore are idempotent so both hook families can
    # safely coexist.
    def on_validation_model_eval(self): self._ema_apply()
    def on_validation_model_train(self): self._ema_restore()
    def on_validation_start(self): self._ema_apply()
    def on_validation_end(self): self._ema_restore()
    def on_test_model_eval(self): self._ema_apply()
    def on_test_model_train(self): self._ema_restore()
    def on_test_start(self): self._ema_apply()
    def on_test_end(self): self._ema_restore()

    def configure_optimizers(self):
        parameters = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if not parameters:
            raise RuntimeError("没有可训练参数")
        learning_rate = float(self.hparams.learning_rate)
        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        warmup_steps = int(self.hparams.warmup_steps)
        try:
            max_steps = int(getattr(self.trainer, "estimated_stepping_batches", 0))
        except (RuntimeError, TypeError, ValueError):
            max_steps = 0
        if max_steps <= 0:
            max_steps = 100000
        eta_min = min(1e-6, learning_rate * 0.01)

        if warmup_steps <= 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, max_steps), eta_min=eta_min)
        elif max_steps <= warmup_steps:
            # There is no post-warmup phase in a very short run.
            scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-3, end_factor=1.0, total_iters=max(1, max_steps)
            )
        else:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps
            )
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, max_steps - warmup_steps), eta_min=eta_min
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
            )
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        clip_val = gradient_clip_val if gradient_clip_val is not None else self.hparams.grad_clip_norm
        if clip_val is not None and float(clip_val) > 0:
            self.clip_gradients(optimizer, gradient_clip_val=float(clip_val), gradient_clip_algorithm=gradient_clip_algorithm or "norm")

    def on_save_checkpoint(self, checkpoint):
        if self.ema: checkpoint["ema_state_dict"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint):
        if self.ema and "ema_state_dict" in checkpoint: self.ema.load_state_dict(checkpoint["ema_state_dict"])

    def on_train_start(self):
        if self.ema:
            self.ema._lazy_init(self.model_sr)
            self._ema_last_step = int(self.global_step)
        print(f"✅ LitRefSRWKV 训练开始 | 参数量: {sum(p.numel() for p in self.model_sr.parameters()) / 1e6:.2f}M | grad_clip={self.hparams.grad_clip_norm} | EMA={'on' if self.ema else 'off'} | SSIM={self.ssim_weight} | FFT={self.fft_weight}")
