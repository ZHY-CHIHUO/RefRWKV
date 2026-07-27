"""
GlobalSemantic.py — 全局语义提取模块 (SR 条件版)

基于 DINOv2 + 双向 RWKV 的金字塔语义聚合器，从参考图像中提取多尺度语义
特征，注入 UNet 各层编码器/瓶颈层。

本版变更：
1. 新增 SR latent 条件分支：SR latent (4×60×60) 经 2×2 avg pool + 自适应
   池化降到 16×16，与 DINOv2 的 256 个 patch token 位置对齐后拼接，
   使金字塔同时看到 "ref 长什么样" 和 "SR 已经重建到什么程度"，
   让语义提取聚焦于 SR 无法提供、必须由 ref 补充的信息。
2. WKV 扫描默认使用纯 PyTorch 分块实现（数值稳定、含 k 与 first/u 的
   标准 RWKV4 语义），保留 wkv_backend="cuda" 旧 kernel 路径供对照。
3. 数值护栏：输入 nan_to_num + clamp，扫描强制 fp32，decay/k/u clamp。

参考文献 (References):
    DINOv2: Oquab et al., "DINOv2: Learning Robust Visual Features without Supervision", 2023
    RWKV:   Peng et al., "RWKV: Reinventing RNNs for the Transformer Era", 2023
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import sys
from pathlib import Path
from typing import Optional

root_dir = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, root_dir)

try:
    from models.RefSRWKV import RUN_CUDA as _RUN_CUDA_NATIVE

    _HAS_CUDA_WKV = True
except (ImportError, ModuleNotFoundError):
    _RUN_CUDA_NATIVE = None
    _HAS_CUDA_WKV = False


# ═══════════════════════════════════════════════════════════════
# 2D 正弦位置编码（供 SR 条件 token 使用）
# ═══════════════════════════════════════════════════════════════


def _get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> torch.Tensor:
    """生成 grid_size × grid_size 的 2D 正弦位置编码。

    Returns:
        (grid_size * grid_size, embed_dim)
    """
    assert embed_dim % 4 == 0, f"embed_dim({embed_dim}) 必须被 4 整除"
    half = embed_dim // 2
    omega = 1.0 / 10000 ** (torch.arange(half // 2).float() / (half // 2))

    gy, gx = torch.meshgrid(
        torch.arange(grid_size).float(),
        torch.arange(grid_size).float(),
        indexing="ij",
    )
    emb_x = gx.flatten().unsqueeze(1) * omega.unsqueeze(0)
    emb_x = torch.cat([emb_x.sin(), emb_x.cos()], dim=1)
    emb_y = gy.flatten().unsqueeze(1) * omega.unsqueeze(0)
    emb_y = torch.cat([emb_y.sin(), emb_y.cos()], dim=1)

    return torch.cat([emb_x, emb_y], dim=1)  # (G*G, embed_dim)


# ═══════════════════════════════════════════════════════════════
# 纯 PyTorch WKV 扫描（分块向量化，fp32，数值稳定）
# ═══════════════════════════════════════════════════════════════


def _wkv_scan_torch(
    decay: torch.Tensor,  # (1, D) 对数衰减（应为负）
    first: torch.Tensor,  # (1, D) 当前步 bonus u
    k: torch.Tensor,  # (B, T, D)
    v: torch.Tensor,  # (B, T, D)
    chunk_size: int = 32,
) -> torch.Tensor:
    """
    标准 RWKV4 WKV（无归一化）：
        wkv[t] = Σ_{i=1}^{t-1} exp((t-1-i)·w + k_i)⊙v_i + exp(u + k_t)⊙v_t

    分块策略：块内构造 (L, L) 衰减矩阵做矩阵乘，块间用状态 S 递推。
    """
    B, T, D = k.shape

    w = decay.float().clamp(min=-12.0, max=0.0)  # log α ≤ 0 → α ∈ (0, 1]
    u = first.float().clamp(min=-30.0, max=20.0)
    k = k.float().clamp(min=-30.0, max=20.0)
    v = v.float()

    S = k.new_zeros(B, D)
    outs = []

    for s in range(0, T, chunk_size):
        L = min(chunk_size, T - s)
        kc = k[:, s : s + L]
        vc = v[:, s : s + L]

        l = torch.arange(L, device=k.device)
        diff = l.view(L, 1) - l.view(1, L)

        logw = diff.view(1, L, L, 1) * w.view(1, 1, 1, D) + kc.unsqueeze(1)
        tril = torch.tril(torch.ones(L, L, device=k.device, dtype=torch.bool))
        logw = logw.masked_fill(~tril.view(1, L, L, 1), -1e9)
        M = torch.exp(logw)

        intra = torch.einsum("bljd,bjd->bld", M, vc)

        decay_l = torch.exp(l.view(1, L, 1) * w.view(1, 1, D))
        intra = intra + decay_l * S.unsqueeze(1)
        outs.append(intra)

        log_end = (L - 1 - l).view(1, L, 1) * w.view(1, 1, D) + kc
        S = torch.exp(L * w) * S + (torch.exp(log_end) * vc).sum(dim=1)

    wkv = torch.cat(outs, dim=1)

    corr = (torch.exp(u) - torch.exp(-w)).view(1, 1, D)
    wkv = wkv + corr * torch.exp(k) * v

    return wkv


def _bi_wkv_scan(decay, first, k, v, backend: str = "torch"):
    """双向 WKV：正向扫描 + 反向扫描取平均。"""
    if backend == "cuda" and _HAS_CUDA_WKV and k.is_cuda:
        v_fwd = _RUN_CUDA_NATIVE(decay, first, k, v)
        v_bwd = _RUN_CUDA_NATIVE(
            decay, first, k.flip(1).contiguous(), v.flip(1).contiguous()
        ).flip(1)
        return 0.5 * (v_fwd + v_bwd)

    v_fwd = _wkv_scan_torch(decay, first, k, v)
    v_bwd = _wkv_scan_torch(decay, first, k.flip(1), v.flip(1)).flip(1)
    return 0.5 * (v_fwd + v_bwd)


# ═══════════════════════════════════════════════════════════════
# SR Latent 条件编码器
# ═══════════════════════════════════════════════════════════════


class SRLatentConditioner(nn.Module):
    """将 SR latent (B, 4, 60, 60) 编码为与 DINOv2 对齐的 patch 特征。

    路径：
        60×60 ──2×2 avg pool──→ 30×30 ──adaptive pool──→ 16×16
        (B, 4, 16, 16) → flatten → (B, 256, 4) → MLP → (B, 256, base_dim)

    与直接对 60×60 做 adaptive pool 相比，先 2×2 平均池化保留更多局部
    纹理统计信息，路径更短、信息更完整。
    """

    def __init__(self, latent_ch: int = 4, base_dim: int = 768, hidden: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(latent_ch, hidden),
            nn.GELU(),
            nn.Linear(hidden, base_dim),
        )
        self.norm = nn.LayerNorm(base_dim)

    def forward(self, sr_latent: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sr_latent: (B, 4, 60, 60) VAE 编码后的 SR latent

        Returns:
            (B, 256, base_dim) 与 DINOv2 patch 网格对齐的条件特征
        """
        x = F.avg_pool2d(sr_latent.float(), kernel_size=2)  # 60→30
        x = F.adaptive_avg_pool2d(x, 16)  # 30→16
        x = x.flatten(2).transpose(1, 2)  # (B, 256, 4)
        return self.norm(self.proj(x))  # (B, 256, D)


# ═══════════════════════════════════════════════════════════════
# RWKV Semantic Aggregator
# ═══════════════════════════════════════════════════════════════


class RWKV_SemanticAggregator(nn.Module):
    """双向 RWKV 语义聚合器（结构与参数名与原版一致，WKV 扫描走稳定后端）。"""

    def __init__(
        self,
        dim: int,
        num_tokens: int = 32,
        hidden_rate: int = 4,
        wkv_backend: str = "torch",
    ):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.wkv_backend = wkv_backend

        self.query = nn.Parameter(torch.empty(1, num_tokens, dim))
        self.query_pos = nn.Parameter(torch.empty(1, num_tokens, dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        nn.init.trunc_normal_(self.query_pos, std=0.02)

        self.input_norm = nn.LayerNorm(dim)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)

        decay = torch.linspace(-1.0, -6.0, dim).unsqueeze(0)
        self.decay = nn.Parameter(decay)
        self.first = nn.Parameter(torch.zeros(1, dim))

        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * hidden_rate),
            nn.GELU(),
            nn.Linear(dim * hidden_rate, dim),
        )

    def forward(self, ref_features: torch.Tensor):
        B, M, C = ref_features.shape

        q = (self.query + self.query_pos).expand(B, -1, -1)
        seq = torch.cat([q, ref_features], dim=1)
        seq = self.input_norm(seq)

        k = self.key(seq)
        v0 = self.value(seq)
        r = torch.sigmoid(self.receptance(seq))

        # WKV 扫描强制 fp32，避免 bf16 下 exp 累积溢出
        with torch.amp.autocast(seq.device.type, enabled=False):
            v = _bi_wkv_scan(self.decay, self.first, k, v0, backend=self.wkv_backend)
        v = v.to(v0.dtype)

        x = self.output(r * v)

        semantic_tokens = x[:, : self.num_tokens, :] + q
        semantic_tokens = semantic_tokens + self.ffn(self.norm(semantic_tokens))
        return semantic_tokens


# ═══════════════════════════════════════════════════════════════
# RWKV Semantic Pyramid
# ═══════════════════════════════════════════════════════════════


class RWKV_SemanticPyramid(nn.Module):
    """多尺度语义金字塔：级联 32 → 16 → 8 → 4 tokens。"""

    def __init__(
        self,
        dim: int,
        token_schedule: tuple = (32, 16, 8, 4),
        level_names: tuple = ("e1", "e2", "e3", "latent"),
        hidden_rate: int = 4,
        use_checkpoint: bool = False,
        wkv_backend: str = "torch",
    ):
        super().__init__()
        if len(token_schedule) != len(level_names):
            raise ValueError(
                f"token_schedule 与 level_names 长度不匹配: "
                f"{len(token_schedule)} vs {len(level_names)}"
            )

        self.level_names = level_names
        self.use_checkpoint = use_checkpoint

        self.aggregators = nn.ModuleList(
            [
                RWKV_SemanticAggregator(
                    dim,
                    num_tokens=n,
                    hidden_rate=hidden_rate,
                    wkv_backend=wkv_backend,
                )
                for n in token_schedule
            ]
        )

    def forward(self, ref_features):
        x = ref_features
        outputs = {}
        for agg, name in zip(self.aggregators, self.level_names):
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(agg, x, use_reentrant=False)
            else:
                x = agg(x)
            outputs[name] = x
        return outputs


# ═══════════════════════════════════════════════════════════════
# Global Semantic Module（SR 条件版）
# ═══════════════════════════════════════════════════════════════


class GlobalSemanticModule(nn.Module):
    """
    全局语义提取器（SR 条件版）

    管线:
        ref ──→ DINOv2 → proj ────────────────┐
                                               ├─ concat ──→ RWKV Pyramid → 输出
        sr_latent ──→ pool/MLP ──→ +pos ──────┘

    SR 条件的作用：
        让金字塔同时看到 ref 的外观和 SR 已重建的结构。SR 已良好的区域
        降低对 ref 的关注，把提取 capacity 留给 SR 缺失、必须由 ref
        补充的纹理/语义信息。

    参数变化（相对原版）：
        新增 use_sr_condition / sr_latent_ch / sr_hidden 三个可选参数，
        其余构造签名、参数名完全不变。use_sr_condition=False 时行为与
        原版等价，旧 checkpoint 可 strict 加载（新增的 sr_* 参数会
        出现在 missing keys 中，需 strict=False 或先冻结加载）。
    """

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        dinov2_model_name: str = "facebook/dinov2-base",
        base_dim: int = 768,
        token_schedule: tuple = (32, 16, 8, 4),
        level_names: tuple = ("e1", "e2", "e3", "latent"),
        hidden_rate: int = 4,
        freeze_dinov2: bool = True,
        use_checkpoint: bool = False,
        wkv_backend: str = "torch",
        use_sr_condition: bool = False,
        sr_latent_ch: int = 4,
        sr_hidden: int = 256,
    ):
        super().__init__()
        self.freeze_dinov2 = freeze_dinov2
        self.use_sr_condition = use_sr_condition

        # ── 1. DINOv2 Backbone ──
        self.dinov2 = AutoModel.from_pretrained(
            dinov2_model_name,
            local_files_only=True,
        )
        dino_dim = self.dinov2.config.hidden_size

        if freeze_dinov2:
            for param in self.dinov2.parameters():
                param.requires_grad = False
            self.dinov2.eval()

        # ── 2. 投影到基础维度 ──
        self.proj = nn.Linear(dino_dim, base_dim)

        # ── 3. SR 条件分支 ──
        if use_sr_condition:
            self.sr_conditioner = SRLatentConditioner(
                latent_ch=sr_latent_ch, base_dim=base_dim, hidden=sr_hidden
            )
            # SR token 的 2D 位置编码（16×16 网格，persistent=False 不入 ckpt）
            sr_pos = _get_2d_sincos_pos_embed(base_dim, 16)
            self.register_buffer("_sr_pos", sr_pos.unsqueeze(0), persistent=False)

        # ── 4. RWKV Semantic Pyramid ──
        self.semantic_pyramid = RWKV_SemanticPyramid(
            dim=base_dim,
            token_schedule=token_schedule,
            level_names=level_names,
            hidden_rate=hidden_rate,
            use_checkpoint=use_checkpoint,
            wkv_backend=wkv_backend,
        )

        self.register_buffer(
            "_dino_mean",
            torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "_dino_std",
            torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1),
        )

    def _normalize_for_dinov2(self, ref_img: torch.Tensor) -> torch.Tensor:
        ref_01 = (ref_img + 1.0) / 2.0
        return (ref_01 - self._dino_mean) / self._dino_std

    def forward(
        self,
        ref_img: torch.Tensor,
        sr_latent: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            ref_img:   (B, 3, H, W) 参考图像，值域 [-1, 1]
            sr_latent: (B, 4, 60, 60) 可选，VAE 编码后的 SR latent。
                       仅在 use_sr_condition=True 时生效。

        Returns:
            dict: {level_name: (B, tokens_i, base_dim), ...}
        """
        # 输入护栏
        ref_img = torch.nan_to_num(ref_img, nan=0.0, posinf=1.0, neginf=-1.0)
        ref_img = ref_img.clamp(-1.5, 1.5)

        if ref_img.shape[-2:] != (224, 224):
            ref_small = F.interpolate(ref_img, size=(224, 224), mode="bilinear")
        else:
            ref_small = ref_img

        ref_small = self._normalize_for_dinov2(ref_small)

        if self.freeze_dinov2:
            with torch.no_grad():
                outputs = self.dinov2(ref_small)
        else:
            outputs = self.dinov2(ref_small)

        features = outputs.last_hidden_state[:, 1:, :]  # (B, 256, dino_dim)
        features = torch.nan_to_num(features, nan=0.0, posinf=10.0, neginf=-10.0)
        features = self.proj(features)  # (B, 256, base_dim)

        # ── SR 条件注入 ──
        if self.use_sr_condition and sr_latent is not None:
            sr_latent = torch.nan_to_num(
                sr_latent, nan=0.0, posinf=20.0, neginf=-20.0
            ).clamp(-20.0, 20.0)
            sr_feats = self.sr_conditioner(sr_latent)  # (B, 256, base_dim)
            sr_feats = sr_feats + self._sr_pos
            combined = torch.cat([features, sr_feats], dim=1)  # (B, 512, D)
        else:
            combined = features

        base_pyramid = self.semantic_pyramid(combined)
        return base_pyramid
