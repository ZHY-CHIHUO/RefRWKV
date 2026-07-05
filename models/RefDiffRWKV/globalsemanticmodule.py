"""
GlobalSemantic.py — 全局语义提取模块 (Global Semantic Extraction Module)

基于 DINOv2 + 双向 RWKV 的金字塔语义聚合器，从参考图像中提取多尺度语义
特征，注入 UNet 各层编码器/瓶颈层，替代 CRefDiff 原有的 LCA_Adapter/SPADE。

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

# ── 项目路径 & CUDA 算子导入 ──
# TODO: 建议在项目入口统一管理 sys.path，避免每个模块重复操作
root_dir = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, root_dir)

try:
    from models.RefSRWKV import RUN_CUDA as _RUN_CUDA_NATIVE

    _HAS_CUDA_WKV = True
except (ImportError, ModuleNotFoundError):
    _RUN_CUDA_NATIVE = None
    _HAS_CUDA_WKV = False


# ═══════════════════════════════════════════════════════════════
# WKV Scan — 带 CUDA/CPU 双后端的时间混合算子
# ═══════════════════════════════════════════════════════════════


def _wkv_forward_scan(
    decay: torch.Tensor,  # (1, dim)
    first: torch.Tensor,  # (1, dim)
    k: torch.Tensor,  # (B, T, dim)
    v: torch.Tensor,  # (B, T, dim)
) -> torch.Tensor:
    """
    RWKV WKV 前向扫描：state[t] = state[t-1] * exp(decay) + v[t].

    CUDA 可用时调用编译算子，否则回退到纯 PyTorch CPU 实现。
    CPU 版本按时间步循环，仅用于调试与单元测试，生产训练请使用 GPU。

    Args:
        decay: (1, dim)  对数衰减因子
        first: (1, dim)  初始状态偏置
        k:     (B, T, dim)  key 序列（CUDA 算子内部使用，CPU 回退暂忽略）
        v:     (B, T, dim)  value 序列

    Returns:
        wkv:   (B, T, dim)  时间混合后状态序列
    """
    if k.is_cuda and _HAS_CUDA_WKV:
        return _RUN_CUDA_NATIVE(decay, first, k, v)

    # ── CPU fallback ──
    B, T, dim = k.shape
    w = torch.exp(decay)  # (1, dim)
    wkv = torch.empty(B, T, dim, device=k.device, dtype=k.dtype)
    state = first.expand(B, dim).clone()  # (B, dim)

    for t in range(T):
        state = state * w + v[:, t, :]
        wkv[:, t, :] = state

    return wkv


# ═══════════════════════════════════════════════════════════════
# RWKV Semantic Aggregator
# ═══════════════════════════════════════════════════════════════


class RWKV_SemanticAggregator(nn.Module):
    """
    RWKV Semantic Aggregator — 双向 RWKV 语义聚合器

    用可学习 Query Token 以 Cross-Attention 风格从 Patch Token 中提取
    固定数量的语义 Token（如 32 → 16 → 8 → 4 逐级压缩）。

    与标准 Transformer Cross-Attention 的区别:
    ┌──────────────────────────────────────────────────────────┐
    │ 标准 Attention:   Q·K^T → softmax → ×V  (O(n²) 复杂度)     │
    │ 双向 RWKV:        time-mixing with learnable decay        │
    │                   + forward/backward scan → avg           │
    │                   (O(n) 线性复杂度)                        │
    └──────────────────────────────────────────────────────────┘

    参数 (Parameters):
        dim:         特征维度
        num_tokens:  输出的语义 Token 数量
        hidden_rate: FFN 隐藏层倍率
    """

    def __init__(self, dim: int, num_tokens: int = 32, hidden_rate: int = 4):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens

        # ── Learnable Query Token & Position Embedding ──
        self.query = nn.Parameter(torch.empty(1, num_tokens, dim))
        self.query_pos = nn.Parameter(torch.empty(1, num_tokens, dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        nn.init.trunc_normal_(self.query_pos, std=0.02)

        # ── RWKV Time-Mixing 组件 ──
        self.input_norm = nn.LayerNorm(dim)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)

        # ── RWKV 衰减参数 ──
        decay = torch.linspace(-1.0, -6.0, dim).unsqueeze(0)
        self.decay = nn.Parameter(decay)
        self.first = nn.Parameter(torch.zeros(1, dim))

        # ── Channel-Mixing (FFN) ──
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * hidden_rate),
            nn.GELU(),
            nn.Linear(dim * hidden_rate, dim),
        )

    def forward(self, ref_features: torch.Tensor):
        """
        Forward pass — 从 Patch Token 中提取语义 Token.

        Args:
            ref_features: (B, M, C) 输入特征，M = Patch 数，C = dim

        Returns:
            semantic_tokens: (B, num_tokens, dim) 提取的语义 Token
        """
        B, M, C = ref_features.shape

        # Step 1: 构建 Query Token + Position
        q = self.query + self.query_pos  # (1, num_tokens, dim)
        q = q.expand(B, -1, -1)  # (B, num_tokens, dim)

        # Step 2: 拼接 [Query, Patch] 序列
        seq = torch.cat([q, ref_features], dim=1)  # (B, num_tokens+M, dim)
        seq = self.input_norm(seq)

        # Step 3: 投影到 K, V, R 空间
        k = self.key(seq)
        v0 = self.value(seq)
        r = torch.sigmoid(self.receptance(seq))

        # Step 4: 双向 RWKV 时间混合
        # 不再除以序列长度 T，保证金字塔各级衰减动力学一致
        v_forward = _wkv_forward_scan(self.decay, self.first, k, v0)

        k_rev = torch.flip(k, dims=[1])
        v_rev_input = torch.flip(v0, dims=[1])
        v_backward = _wkv_forward_scan(self.decay, self.first, k_rev, v_rev_input)
        v_backward = torch.flip(v_backward, dims=[1])

        v = 0.5 * (v_forward + v_backward)

        # Step 5: 门控输出
        x = r * v
        x = self.output(x)

        # Step 6: 取出 Query 部分 + Residual + FFN
        semantic_tokens = x[:, : self.num_tokens, :] + q
        semantic_tokens = semantic_tokens + self.ffn(self.norm(semantic_tokens))

        return semantic_tokens  # (B, num_tokens, dim)


# ═══════════════════════════════════════════════════════════════
# RWKV Semantic Pyramid
# ═══════════════════════════════════════════════════════════════


class RWKV_SemanticPyramid(nn.Module):
    """
    RWKV Semantic Pyramid — 多尺度语义金字塔

    级联多层 RWKV_SemanticAggregator，逐级压缩语义 Token 数量。

    默认 schedule: 256 patches → 32 → 16 → 8 → 4 tokens
    对应 UNet 编码器的不同层级:
        tokens[0] (32) → e1     (down_block[0], 最高分辨率)
        tokens[1] (16) → e2     (down_block[1])
        tokens[2] (8)  → e3     (down_block[2])
        tokens[3] (4)  → latent (mid_block,   最低分辨率)

    设计理念:
        高层级保留更多空间细节 → 对分辨率敏感的浅层有益
        低层级保留全局语义信息 → 对语义敏感的瓶颈层有益

    参数 (Parameters):
        dim:              特征维度
        token_schedule:   每级输出的 Token 数量，如 (32, 16, 8, 4)
        level_names:      每级对应的名称，如 ("e1", "e2", "e3", "latent")
        hidden_rate:      FFN 隐藏层倍率
        use_checkpoint:   是否对聚合器序列使用梯度检查点
    """

    def __init__(
        self,
        dim: int,
        token_schedule: tuple = (32, 16, 8, 4),
        level_names: tuple = ("e1", "e2", "e3", "latent"),
        hidden_rate: int = 4,
        use_checkpoint: bool = False,
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
                RWKV_SemanticAggregator(dim, num_tokens=n, hidden_rate=hidden_rate)
                for n in token_schedule
            ]
        )

    def forward(self, ref_features):
        """
        Args:
            ref_features: (B, N_in, base_dim) 输入特征

        Returns:
            dict: {level_name: (B, tokens_i, base_dim), ...}
        """
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
# Global Semantic Module
# ═══════════════════════════════════════════════════════════════


class GlobalSemanticModule(nn.Module):
    """
    Global Semantic Module — 全局语义提取器

    完整的 ref 语义提取管线:
        DINOv2 → 投影 → RWKV Semantic Pyramid → 输出 base_dim token
        （由 sd2_control_ldm.sem_proj 统一投影到 cross_attn_dim）

    DINOv2 的作用:
        提供预训练的、鲁棒的视觉特征。DINOv2 通过自监督训练在
        1.42 亿张图上学习，其 Patch Token 天然包含丰富的语义和
        空间信息，是理想的特征提取 Backbone。

    RWKV Pyramid 的作用:
        将 DINOv2 Patch Token 逐级压缩为少量语义 Token，
        用线性复杂度的双向 RWKV 替代二次复杂度的 Cross-Attention，
        同时保持全局感受野。

    参数 (Parameters):
        dinov2_model_name: DINOv2 模型名 (自动读取 hidden_size)
            - "facebook/dinov2-small"  → hidden_size=384
            - "facebook/dinov2-base"   → hidden_size=768  (默认)
            - "facebook/dinov2-large"  → hidden_size=1024
            - "facebook/dinov2-giant"  → hidden_size=1536
        base_dim:          金字塔内部工作维度 (默认 128，原为 64)
                           增大可缓解 DINOv2→金字塔的信息瓶颈，
                           代价是 RWKV 计算量按 O(base_dim²) 增长
                           后续层自动按 ×1, ×2, ×4, ×8 推导
        token_schedule:    金字塔每级 Token 数，默认 (32, 16, 8, 4)
        level_names:       对应的层级名称，默认 ("e1", "e2", "e3", "latent")
        hidden_rate:       RWKV FFN 隐藏层倍率
        freeze_dinov2:     是否冻结 DINOv2 (默认 True，只做特征提取)
        use_checkpoint:    是否在金字塔中使用梯度检查点
    """

    # ImageNet 归一化常量（DINOv2 预训练时使用）
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
    ):
        super().__init__()
        self.freeze_dinov2 = freeze_dinov2

        # ── 1. DINOv2 Backbone ──
        self.dinov2 = AutoModel.from_pretrained(
            dinov2_model_name,
            local_files_only=True,
        )
        dino_dim = self.dinov2.config.hidden_size

        if freeze_dinov2:
            for param in self.dinov2.parameters():
                param.requires_grad = False
            self.dinov2.eval()  # 冻结时关闭 dropout，保证确定性输出

        # ── 2. 投影到基础维度 ──
        self.proj = nn.Linear(dino_dim, base_dim)

        # ── 3. RWKV Semantic Pyramid ──
        self.semantic_pyramid = RWKV_SemanticPyramid(
            dim=base_dim,
            token_schedule=token_schedule,
            level_names=level_names,
            hidden_rate=hidden_rate,
            use_checkpoint=use_checkpoint,
        )


    def _normalize_for_dinov2(self, ref_img: torch.Tensor) -> torch.Tensor:
        """
        对输入图像应用 ImageNet 归一化，使其符合 DINOv2 预训练分布。

        自动检测输入值域:
            - 若存在负值 → 假定 [-1, 1]，先映射到 [0, 1]
            - 若全为非负 → 假定已是 [0, 1]，直接归一化

        Args:
            ref_img: (B, 3, H, W)，值域 [-1, 1] 或 [0, 1]

        Returns:
            normalized: (B, 3, H, W)，经 ImageNet 均值/标准差归一化
        """
        if ref_img.min() < 0:
            ref_img = (ref_img + 1) / 2  # [-1, 1] → [0, 1]

        mean = torch.tensor(
            self.IMAGENET_MEAN, device=ref_img.device, dtype=ref_img.dtype
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            self.IMAGENET_STD, device=ref_img.device, dtype=ref_img.dtype
        ).view(1, 3, 1, 1)

        return (ref_img - mean) / std

    def forward(self, ref_img: torch.Tensor):
        """
        Forward pass — 从参考图像提取多尺度语义特征.

        Args:
            ref_img: (B, 3, H, W) 参考图像，值域 [-1, 1]（推荐）或 [0, 1]
                     会被 resize 到 224×224 以匹配 DINOv2 输入

        Returns:
            dict: {
                level_name: (B, tokens_i, base_dim),  # 如 (B, 32, 128)
                ...
            }
            由调用方负责投影到目标维度。
        """
        # ── Step 1: Resize to DINOv2 fixed input size ──
        if ref_img.shape[-2:] != (224, 224):
            ref_small = F.interpolate(ref_img, size=(224, 224), mode="bilinear")
        else:
            ref_small = ref_img

        # ── Step 2: ImageNet 归一化 ──
        ref_small = self._normalize_for_dinov2(ref_small)

        # ── Step 3: DINOv2 特征提取 ──
        if self.freeze_dinov2:
            with torch.no_grad():
                outputs = self.dinov2(ref_small)
        else:
            outputs = self.dinov2(ref_small)

        # 去掉 CLS token (index 0)，只保留 Patch tokens (index 1..257)
        features = outputs.last_hidden_state[:, 1:, :]  # (B, 256, dino_dim)

        # ── Step 4: 投影到基础维度 ──
        features = self.proj(features)  # (B, 256, base_dim)

        # ── Step 5: RWKV Semantic Pyramid ──
        base_pyramid = self.semantic_pyramid(features)

        # ── Step 6: 直接返回 base_dim token，由调用方投影 ──
        return base_pyramid




