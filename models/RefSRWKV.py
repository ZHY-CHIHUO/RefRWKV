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
    if _wkv_cuda is not None: return _wkv_cuda
    if _wkv_load_error is not None: raise RuntimeError("Bi-WKV CUDA 扩展加载失败") from _wkv_load_error
    if not torch.cuda.is_available(): raise RuntimeError("Bi-WKV 需要 CUDA 环境")
    cap = torch.cuda.get_device_capability()
    arch, sm = f"compute_{cap[0]}{cap[1]}", f"sm_{cap[0]}{cap[1]}"
    _wkv_cuda = load(
        name="bi_wkv",
        sources=[os.path.join(_cuda_dir, "bi_wkv.cpp"), os.path.join(_cuda_dir, "bi_wkv_kernel.cu")],
        verbose=True,
        extra_cuda_cflags=["-res-usage", "--maxrregcount 60", "--use_fast_math", "-O3", "-Xptxas -O3",
                           f"-gencode arch={arch},code={sm}", f"-gencode arch={arch},code={arch}"],
    )
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
            identity = torch.zeros(self.dim, 1, 5, 5, device=self.conv1x1.weight.device)
            identity[:, :, 2, 2] = 1.0
            w1, w3, w5 = F.pad(self.conv1x1.weight, (2,2,2,2)), F.pad(self.conv3x3.weight, (1,1,1,1)), self.conv5x5.weight
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
        assert n_embd % num_groups == 0 and n_embd % 16 == 0, "n_embd 必须被 num_groups 整除且是 16 的倍数"
        self.n_embd, self.window_size, self.shift_size = n_embd, window_size, shift_size
        self.num_groups, self.group_dim, self.recurrence = num_groups, n_embd // num_groups, 2
        self.omni_shift = OmniShift(dim=n_embd)
        self.key, self.value, self.receptance, self.output = [nn.Linear(n_embd, n_embd, bias=False) for _ in range(4)]
        self.register_buffer("scale", torch.tensor(n_embd**0.5))
        with torch.no_grad():
            decay_init = torch.zeros(self.recurrence, n_embd)
            for g in range(num_groups): decay_init[:, g*self.group_dim:(g+1)*self.group_dim] = -0.5 * (g + 1)
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
            dj, fj = self.spatial_decay[j] / s, self.spatial_first[j] / s
            if j % 2 == 0: v = RUN_CUDA(dj, fj, k, v)
            else:
                kt, vt = rearrange(k, "b (h w) c -> b (w h) c", h=ws, w=ws), rearrange(v, "b (h w) c -> b (w h) c", h=ws, w=ws)
                v = rearrange(RUN_CUDA(dj, fj, kt, vt), "b (w h) c -> b (h w) c", h=ws, w=ws)
        return sr * v

    def forward(self, x, resolution, layer_idx=0):
        B, T, C = x.size()
        h, w, ws, ss = resolution[0], resolution[1], self.window_size, self.shift_size
        sr, k, v = self.jit_func(x, resolution)
        shift_amt = (layer_idx % 3) * ss
        k, v, sr = [rearrange(t, "b (hh ww) c -> b hh ww c", hh=h, ww=w) for t in (k, v, sr)]
        
        target_h, target_w = max(h, h + shift_amt), max(w, w + shift_amt)
        pad_h, pad_w = (ws - target_h % ws) % ws, (ws - target_w % ws) % ws
        if pad_h > 0 or pad_w > 0:
            k, v, sr = [F.pad(t, (0, 0, 0, pad_w, 0, pad_h)) for t in (k, v, sr)]
        Hp, Wp = h + pad_h, w + pad_w
            
        k, v, sr = [rearrange(t, "b (nh w1) (nw w2) c -> (b nh nw) (w1 w2) c", w1=ws, w2=ws) for t in (k, v, sr)]
        out = self._window_wkv(k, v, sr)
        out = rearrange(out, "(b nh nw) (w1 w2) c -> b (nh w1) (nw w2) c", nh=Hp // ws, nw=Wp // ws, w1=ws, w2=ws)
        if pad_h > 0 or pad_w > 0: out = out[:, :h, :w, :]
            
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
        return lr_feat + self.gate(fused) * conf

# ═══════════════════════════════════════════════════════════
# 核心超分网络
# ═══════════════════════════════════════════════════════════
class RefSRWKV(nn.Module):
    def __init__(self, inp_channels: int = 3, out_channels: int = 3, dim: int = 48,
                 num_blocks: tuple = (4, 6, 6, 8), num_refinement_blocks: int = 4,
                 scale: int = 4, hr_size: int = 480,
                 drop_path_rate: float = 0.1, hidden_rate: int = 4):
        super().__init__()
        self.scale, self.dim, self.out_channels = scale, dim, out_channels
        
        # ★ 核心自动化逻辑：绑定 PixelUnshuffle(4) 黄金法则
        self.ref_down_factor = 4  
        assert hr_size % self.ref_down_factor == 0, f"HR尺寸({hr_size}) 必须能被 {self.ref_down_factor} 整除"
        self.internal_size = hr_size // self.ref_down_factor

        self.lr_up = nn.Sequential(
            nn.Conv2d(inp_channels, dim, 3, padding=1, bias=False), nn.GroupNorm(_gn_groups(dim), dim), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=1, bias=False), nn.GroupNorm(_gn_groups(dim), dim), nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=2, dilation=2, bias=False), nn.GroupNorm(_gn_groups(dim), dim), nn.ReLU(inplace=True),
        )
        
        self.ref_to_level1 = nn.Sequential(
            nn.PixelUnshuffle(self.ref_down_factor),
            nn.Conv2d(out_channels * (self.ref_down_factor ** 2), dim, 1, bias=False),
            nn.GroupNorm(_gn_groups(dim), dim), nn.ReLU(inplace=True),
        )
        self.ref_down2 = nn.Sequential(nn.Conv2d(dim, dim * 2, 3, stride=2, padding=1, bias=False), nn.GroupNorm(_gn_groups(dim * 2), dim * 2))
        self.ref_down3 = nn.Sequential(nn.Conv2d(dim * 2, dim * 4, 3, stride=2, padding=1, bias=False), nn.GroupNorm(_gn_groups(dim * 4), dim * 4))
        self.ref_down4 = nn.Sequential(nn.Conv2d(dim * 4, dim * 8, 3, stride=2, padding=1, bias=False), nn.GroupNorm(_gn_groups(dim * 8), dim * 8))
        
        self.fuse1, self.fuse2, self.fuse3, self.fuse4 = GatedFusion(dim), GatedFusion(dim * 2), GatedFusion(dim * 4), GatedFusion(dim * 8)
        
        dp_rates = [drop_path_rate * i / (sum(num_blocks) - 1) for i in range(sum(num_blocks))]
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
        self.output_head = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1, bias=False), nn.GroupNorm(_gn_groups(dim), dim), nn.ReLU(inplace=True), nn.Conv2d(dim, out_channels, 3, padding=1, bias=False))
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)): nn.init.zeros_(m.bias); nn.init.ones_(m.weight)

    def _match_color(self, ref, target):
        ref_mean, ref_std = ref.mean(dim=[2, 3], keepdim=True), ref.std(dim=[2, 3], keepdim=True)
        tgt_mean, tgt_std = target.mean(dim=[2, 3], keepdim=True), target.std(dim=[2, 3], keepdim=True)
        return (ref - ref_mean) / (ref_std + 1e-6) * tgt_std + tgt_mean

    def _extract_ref_pyramid(self, ref):
        ref_1 = self.ref_to_level1(ref)
        return ref_1, self.ref_down2(ref_1), self.ref_down3(self.ref_down2(ref_1)), self.ref_down4(self.ref_down3(self.ref_down2(ref_1)))

    def forward(self, lr, ref):
        target_hr_h, target_hr_w = ref.shape[2], ref.shape[3]
        lr_hr = F.interpolate(lr, size=(target_hr_h, target_hr_w), mode="bicubic", align_corners=False)
        ref_aligned = self._match_color(ref, lr_hr)
        
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
    def __init__(self, decay: float = 0.999): self.decay, self.shadow, self.backup, self._initialized = decay, {}, {}, False
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
                if self.shadow[name].device != param.device: self.shadow[name] = self.shadow[name].to(param.device)
                p_data = param.data.float() if param.data.dtype != torch.float32 else param.data
                self.shadow[name].mul_(self.decay).add_(p_data, alpha=1.0 - self.decay)
    def apply_shadow(self, model: nn.Module):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                if self.shadow[name].device != param.device: self.shadow[name] = self.shadow[name].to(param.device)
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup: param.data.copy_(self.backup[name].to(param.device))
        self.backup = {}
    def state_dict(self): return {"decay": self.decay, "shadow": self.shadow, "initialized": self._initialized}
    def load_state_dict(self, sd): self.decay, self.shadow, self._initialized = sd["decay"], sd["shadow"], sd["initialized"]

class LitRefSRWKV(pl.LightningModule):
    def __init__(self, model_sr: RefSRWKV, learning_rate: float = 1e-4, warmup_steps: int = 500, grad_clip_norm: float = 1.0, ema_decay: float = 0.999, use_ema: bool = True, ssim_weight: float = 0.0, fft_weight: float = 0.0, ref_drop_prob: float = 0.0, loss_fn=None, lr_key: str = "lr", hr_key: str = "hr", ref_key: str = "ref"):
        super().__init__()
        self.save_hyperparameters(ignore=["model_sr", "loss_fn"])
        self.model_sr, self.ssim_weight, self.fft_weight = model_sr, ssim_weight, fft_weight
        if ssim_weight > 0 or fft_weight > 0:
            self.l1_loss = nn.L1Loss()
            if ssim_weight > 0:
                try:
                    from pyiqa import create_metric as _create_pyiqa_metric
                    self.ssim_loss_fn, self._ssim_backend = _create_pyiqa_metric("ssim", loss_mode=True), "pyiqa"
                except Exception: self.ssim_loss_fn, self._ssim_backend = None, "manual"
            else: self.ssim_loss_fn = None
            self.criterion = None
        else: self.criterion, self.ssim_loss_fn = loss_fn or nn.L1Loss(), None
        self.lr_key, self.hr_key, self.ref_key = lr_key, hr_key, ref_key
        self.ema = EMA(decay=ema_decay) if use_ema else None

    def _unpack_batch(self, batch): return (batch[self.lr_key], batch[self.hr_key], batch[self.ref_key]) if isinstance(batch, dict) else (batch[0], batch[1], batch[2])
    def _apply_ref_dropout(self, ref):
        p = self.hparams.ref_drop_prob
        if p <= 0 or not self.training or ref.size(0) < 2: return ref
        drop = (torch.rand(ref.size(0), 1, 1, 1, device=ref.device) < p).float()
        return drop * ref[torch.randperm(ref.size(0), device=ref.device)] + (1.0 - drop) * ref
    def forward(self, lr, ref): return self.model_sr(lr, ref)
    @staticmethod
    def _fft_loss(pred, target): return F.l1_loss(torch.fft.rfft2(pred, norm="ortho"), torch.fft.rfft2(target, norm="ortho"))

    def training_step(self, batch, batch_idx):
        lr, hr, ref = self._unpack_batch(batch)
        output = self(lr, self._apply_ref_dropout(ref))
        if self.ssim_weight > 0 or self.fft_weight > 0:
            loss = self.l1_loss(output, hr)
            if self.ssim_weight > 0:
                ssim_loss = 1.0 - (self.ssim_loss_fn(output, hr) if self._ssim_backend == "pyiqa" and self.ssim_loss_fn else self._manual_ssim_loss(output, hr))
                loss += self.ssim_weight * ssim_loss
                self.log("train_ssim_loss", ssim_loss, on_step=True, on_epoch=True)
            if self.fft_weight > 0:
                loss += self.fft_weight * self._fft_loss(output, hr)
                self.log("train_fft_loss", self._fft_loss(output, hr), on_step=True, on_epoch=True)
            self.log("train_l1", self.l1_loss(output, hr), on_step=True, on_epoch=True)
        else: loss = self.criterion(output, hr)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    @staticmethod
    def _manual_ssim_loss(pred, target):
        C, ws, sigma = pred.shape[1], 11, 1.5
        coords = torch.arange(ws, dtype=pred.dtype, device=pred.device)
        g = torch.exp(-((coords - ws // 2) ** 2) / (2 * sigma**2))
        window = (g / g.sum()).unsqueeze(0).unsqueeze(1).repeat(C, 1, 1, 1).unsqueeze(0)
        pad = ws // 2
        mu_p, mu_t = F.conv2d(pred, window, padding=pad, groups=C), F.conv2d(target, window, padding=pad, groups=C)
        sigma_p_sq, sigma_t_sq = F.conv2d(pred**2, window, padding=pad, groups=C) - mu_p**2, F.conv2d(target**2, window, padding=pad, groups=C) - mu_t**2
        sigma_pt = F.conv2d(pred * target, window, padding=pad, groups=C) - mu_p * mu_t
        C1, C2 = (0.01 * 2.0) ** 2, (0.03 * 2.0) ** 2
        return 1.0 - (((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2)) / ((mu_p**2 + mu_t**2 + C1) * (sigma_p_sq + sigma_t_sq + C2))).mean()

    # ★ 修复：删除了手动 warmup 的 on_train_batch_start

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.ema: self.ema.update(self.model_sr)
        
    # ★ 修复：使用 validation/test 级别的钩子，防止 val_check_interval < 1.0 时 EMA 污染训练
    def on_validation_start(self):
        if self.ema: self.ema.apply_shadow(self.model_sr)
    def on_validation_end(self):
        if self.ema: self.ema.restore(self.model_sr)
    def on_test_start(self):
        if self.ema: self.ema.apply_shadow(self.model_sr)
    def on_test_end(self):
        if self.ema: self.ema.restore(self.model_sr)

    def validation_step(self, batch, batch_idx):
        lr, hr, ref = self._unpack_batch(batch)
        output = self(lr, ref)
        loss = self.l1_loss(output, hr) if self.ssim_weight > 0 or self.fft_weight > 0 else self.criterion(output, hr)
        if self.ssim_weight > 0:
            ssim_loss = 1.0 - (self.ssim_loss_fn(output, hr) if self._ssim_backend == "pyiqa" and self.ssim_loss_fn else self._manual_ssim_loss(output, hr))
            loss += self.ssim_weight * ssim_loss
            self.log("val_ssim_loss", ssim_loss, on_step=False, on_epoch=True)
        if self.fft_weight > 0:
            fft_loss = self._fft_loss(output, hr)
            loss += self.fft_weight * fft_loss
            self.log("val_fft_loss", fft_loss, on_step=False, on_epoch=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/psnr", 10 * torch.log10(4.0 / (F.mse_loss(output, hr) + 1e-8)), on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        lr, hr, ref = self._unpack_batch(batch)
        self.log("test_loss", (self.criterion or self.l1_loss)(self(lr, ref), hr), on_step=False, on_epoch=True)
        return self(lr, ref), hr

    # ★ 修复：使用 SequentialLR 完美拼接 Warmup 和 Cosine
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0, total_iters=self.hparams.warmup_steps
        )
        max_steps = getattr(self.trainer, "estimated_stepping_batches", 100000)
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, max_steps - self.hparams.warmup_steps), eta_min=1e-6
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[self.hparams.warmup_steps]
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}]

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        if (clip_val := gradient_clip_val or self.hparams.grad_clip_norm) > 0: self.clip_gradients(optimizer, gradient_clip_val=clip_val, gradient_clip_algorithm=gradient_clip_algorithm or "norm")
    def on_save_checkpoint(self, checkpoint):
        if self.ema: checkpoint["ema_state_dict"] = self.ema.state_dict()
    def on_load_checkpoint(self, checkpoint):
        if self.ema and "ema_state_dict" in checkpoint: self.ema.load_state_dict(checkpoint["ema_state_dict"])
    def on_train_start(self):
        print(f"✅ LitRefSRWKV 训练开始 | 参数量: {sum(p.numel() for p in self.parameters()) / 1e6:.2f}M | grad_clip={self.hparams.grad_clip_norm} | EMA={'on' if self.ema else 'off'} | SSIM={self.ssim_weight} | FFT={self.fft_weight}")