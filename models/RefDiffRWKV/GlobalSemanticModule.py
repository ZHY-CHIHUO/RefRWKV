"""
GlobalSemantic.py — 全局语义提取模块 (Global Semantic Extraction Module)

基于 DINOv2 + 双向 RWKV 的金字塔语义聚合器，从参考图像中提取多尺度语义
特征，注入 UNet 各层编码器/瓶颈层，替代 CRefDiff 原有的 LCA_Adapter/SPADE。

核心管线 (Pipeline):
    Ref Image (224×224)
        │
        ▼
    DINOv2 (冻结) ──► (B, 256, dino_dim) Patch Tokens
        │
        ▼
    Linear Proj ────► (B, 256, base_dim)   统一内部维度
        │
        ▼
    RWKV Semantic Pyramid:
        agg32 → sem32 (B, 32, base_dim)
           │
        agg16 → sem16 (B, 16, base_dim)
           │
        agg8  → sem8  (B, 8,  base_dim)
           │
        agg4  → sem4  (B, 4,  base_dim)
        │
        ▼
    Level Proj ─────► 投影到 UNet 各层实际通道数:
        e1: base_dim → unet_dim       (如 64→320)
        e2: base_dim → unet_dim×2     (如 64→640)
        e3: base_dim → unet_dim×4     (如 64→1280)
        latent: base_dim → unet_dim×8 (如 64→2560)

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
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# Add project root to sys.path so we can import RUN_CUDA
# 将项目根目录加入 sys.path，以便导入 RUN_CUDA（双向 WKV CUDA 算子）
root_dir = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, root_dir)
from models.RefSRWKV import RUN_CUDA


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
        # 可学习的查询 Token 和位置编码
        # query:   (1, num_tokens, dim) → 被广播到 batch 维度
        # query_pos: (1, num_tokens, dim) → 与 query 相加提供位置先验
        self.query = nn.Parameter(torch.empty(1, num_tokens, dim))
        self.query_pos = nn.Parameter(torch.empty(1, num_tokens, dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        nn.init.trunc_normal_(self.query_pos, std=0.02)

        # ── RWKV Time-Mixing 组件 ──
        # Key / Value / Receptance / Output projection
        # 键 / 值 / 接受度 / 输出投影
        self.input_norm = nn.LayerNorm(dim)  # 输入预归一化 (pre-norm)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)  # 门控接受度 (sigmoid gating)
        self.output = nn.Linear(dim, dim, bias=False)

        # ── RWKV 衰减参数 ──
        # decay: (1, dim) → 控制时间混合的衰减速率
        # first: (1, dim) → 第一个位置的特殊偏置
        decay = torch.linspace(-1.0, -6.0, dim).unsqueeze(0)
        self.decay = nn.Parameter(decay)
        self.first = nn.Parameter(torch.zeros(1, dim))

        # ── Channel-Mixing (FFN) ──
        # 通道混合：LayerNorm → Linear↑ → GELU → Linear↓
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

        流程 (Workflow):
            1. [QueryToken | PatchToken] → 拼接
            2. LayerNorm → Key/Value/Receptance → 双向 WKV → Output
            3. 取出 Query 部分 → Residual + FFN → 输出
        """
        B, M, C = ref_features.shape

        # Step 1: 构建 Query Token + Position (广播到 batch)
        q = self.query + self.query_pos  # (1, num_tokens, dim)
        q = q.expand(B, -1, -1)  # (B, num_tokens, dim)

        # Step 2: 拼接 [Query, Patch]，构成完整输入序列
        # Concatenate: learnable queries first, followed by all patch tokens
        # 序列结构: [q1 q2 ... qN | patch1 patch2 ... patchM]
        seq = torch.cat([q, ref_features], dim=1)  # (B, num_tokens+M, dim)
        seq = self.input_norm(seq)
        T = seq.shape[1]

        # Step 3: 投影到 K, V, R 空间
        # Project to Key, Value, Receptance spaces
        k = self.key(seq)  # (B, T, dim)
        v0 = self.value(seq)  # (B, T, dim)
        r = torch.sigmoid(self.receptance(seq))  # (B, T, dim), 门控 0~1

        # Step 4: 双向 RWKV 时间混合
        # Bidirectional RWKV time-mixing:
        #   - Forward scan:  从左到右，每个 token 聚合历史信息
        #   - Backward scan: 从右到左，每个 token 聚合未来信息
        #   - 最终取平均，达成双向感受野

        # Forward direction (前向扫描)
        v_forward = RUN_CUDA(
            self.decay / T,  # 衰减按序列长度归一化
            self.first / T,  # 首位置偏置同样归一化
            k,
            v0,
        )  # (B, T, dim)

        # Backward direction (反向扫描 → flip → WKV → flip back)
        k_rev = torch.flip(k, dims=[1])  # 翻转 key 序列
        v_rev_input = torch.flip(v0, dims=[1])  # 翻转 value 序列
        v_backward = RUN_CUDA(
            self.decay / T, self.first / T, k_rev, v_rev_input
        )  # (B, T, dim)
        v_backward = torch.flip(v_backward, dims=[1])  # 翻转回原始顺序

        # Average forward + backward → 双向感受野
        # 前向+反向取平均，确保每个 Query Token 同时看到左右两侧的 Patch
        v = 0.5 * (v_forward + v_backward)  # (B, T, dim)

        # Step 5: 门控输出
        # Gate with receptance: x = r ⊙ v (element-wise)
        x = r * v  # (B, T, dim)
        x = self.output(x)  # (B, T, dim)

        # Step 6: 取出 Query 部分 + Residual + FFN
        # Extract only the first num_tokens (the Query positions)
        # 只取序列前 num_tokens 个位置（Query Token 部分）
        # 丢弃后面的 Patch Token（它们只是上下文，不参与输出）
        semantic_tokens = x[:, : self.num_tokens, :] + q  # Residual with original query
        semantic_tokens = semantic_tokens + self.ffn(
            self.norm(semantic_tokens)
        )  # Pre-norm FFN

        return semantic_tokens  # (B, num_tokens, dim)


class RWKV_SemanticPyramid(nn.Module):
    """
    RWKV Semantic Pyramid — 多尺度语义金字塔

    级联 4 层 RWKV_SemanticAggregator，逐级压缩语义 Token 数量:
        256 patches → agg32 → 32 tokens → agg16 → 16 tokens
                  → agg8 → 8 tokens → agg4 → 4 tokens

    每一级的输出对应 UNet 编码器的不同层级:
        sem32 → e1     (down_block[0], 最高分辨率)
        sem16 → e2     (down_block[1])
        sem8  → e3     (down_block[2])
        sem4  → latent (mid_block,   最低分辨率)

    设计理念:
        高层级（sem32）保留更多空间细节 → 对分辨率敏感的浅层有益
        低层级（sem4） 保留全局语义信息 → 对语义敏感的瓶颈层有益
    """

    def __init__(self, dim: int, hidden_rate: int = 4):
        super().__init__()
        # 四级聚合器，逐级 Token 减半
        self.agg32 = RWKV_SemanticAggregator(
            dim, num_tokens=32, hidden_rate=hidden_rate
        )  # 256 → 32
        self.agg16 = RWKV_SemanticAggregator(
            dim, num_tokens=16, hidden_rate=hidden_rate
        )  # 32 → 16
        self.agg8 = RWKV_SemanticAggregator(
            dim, num_tokens=8, hidden_rate=hidden_rate
        )  # 16 → 8
        self.agg4 = RWKV_SemanticAggregator(
            dim, num_tokens=4, hidden_rate=hidden_rate
        )  # 8 → 4

    def forward(self, ref_features):
        """
        Args:
            ref_features: (B, 256, base_dim) DINOv2 Patch Token 投影后的特征

        Returns:
            dict: {
                "e1":     (B, 32, base_dim),
                "e2":     (B, 16, base_dim),
                "e3":     (B, 8,  base_dim),
                "latent": (B, 4,  base_dim),
            }
        """
        sem32 = self.agg32(ref_features)  # 256 → 32 tokens
        sem16 = self.agg16(sem32)  # 32  → 16 tokens
        sem8 = self.agg8(sem16)  # 16  → 8  tokens
        sem4 = self.agg4(sem8)  # 8   → 4  tokens
        return {
            "e1": sem32,  # 浅层: 32 tokens, 高空间精度
            "e2": sem16,  # 中层: 16 tokens
            "e3": sem8,  # 深层: 8  tokens
            "latent": sem4,  # 瓶颈: 4  tokens, 全局语义
        }


class GlobalSemanticModule(nn.Module):
    """
    Global Semantic Module — 全局语义提取器

    完整的 ref 语义提取管线:
        DINOv2 → 投影 → RWKV Semantic Pyramid → UNet 层级维度投影

    DINOv2 的作用:
        提供预训练的、鲁棒的视觉特征。DINOv2 通过自监督训练在
        1.42 亿张图上学习，其 Patch Token 天然包含丰富的语义和
        空间信息，是理想的特征提取 Backbone。

    RWKV Pyramid 的作用:
        将 256 个 DINOv2 Patch Token 逐级压缩为 32→16→8→4 个
        语义 Token，用线性复杂度的双向 RWKV 替代二次复杂度的
        Cross-Attention，同时保持全局感受野。

    参数 (Parameters):
        dinov2_model_name: DINOv2 模型名 (自动读取 hidden_size)
            - "facebook/dinov2-small"  → hidden_size=384
            - "facebook/dinov2-base"   → hidden_size=768
            - "facebook/dinov2-large"  → hidden_size=1024
            - "facebook/dinov2-giant"  → hidden_size=1536
        base_dim:          金字塔内部工作维度 (统一处理空间)
        unet_dim:          UNet 第一层的 embed_dim
                           后续层自动按 ×1, ×2, ×4, ×8 推导
        hidden_rate:       RWKV FFN 隐藏层倍率
        freeze_dinov2:     是否冻结 DINOv2 (默认 True，只做特征提取)
    """

    def __init__(
        self,
        dinov2_model_name: str = "facebook/dinov2-base",
        base_dim: int = 64,
        unet_dim: int = 384,
        hidden_rate: int = 4,
        freeze_dinov2: bool = True,
    ):
        super().__init__()
        self.freeze_dinov2 = freeze_dinov2

        # ── 1. DINOv2 Backbone ──
        # 自动检测 hidden_size，兼容所有 DINOv2 变体
        self.dinov2 = AutoModel.from_pretrained(
            dinov2_model_name,
            local_files_only=True,
        )
        dino_dim = self.dinov2.config.hidden_size  # 768 (base) / 384 (small) / ...
        if freeze_dinov2:
            # 冻结 DINOv2：只读特征提取，不参与梯度更新
            for param in self.dinov2.parameters():
                param.requires_grad = False

        # ── 2. 投影到基础维度 ──
        # Project DINOv2 dim → internal base_dim
        # 将 DINOv2 的高维特征（768/384/...）压缩到统一的基础维度（如 64）
        # 这样做的好处：
        #   a) RWKV Pyramid 的计算量与 base_dim² 成正比，小 dim 更快
        #   b) 解耦 Backbone 与下游模块，方便更换 Backbone
        self.proj = nn.Linear(dino_dim, base_dim)

        # ── 3. RWKV Semantic Pyramid ──
        # 4 级双向 RWKV 聚合器：256 patches → 32 → 16 → 8 → 4 tokens
        self.semantic_pyramid = RWKV_SemanticPyramid(
            dim=base_dim, hidden_rate=hidden_rate
        )

        # ── 4. 层级维度投影 ──
        # 金字塔各层输出维度均为 (B, N, base_dim)
        # 需要投影到 UNet 各层的实际通道数:
        #   e1:     base_dim → unet_dim       (如 64→320)
        #   e2:     base_dim → unet_dim×2     (如 64→640)
        #   e3:     base_dim → unet_dim×4     (如 64→1280)
        #   latent: base_dim → unet_dim×8     (如 64→2560)
        dim_map = {
            "e1": unet_dim,  # down_blocks[0] → 320 ch
            "e2": unet_dim * 2,  # down_blocks[1] → 640 ch
            "e3": unet_dim * 4,  # down_blocks[2] → 1280 ch
            "latent": unet_dim * 8,  # mid_block     → 2560 ch (if unet_dim=320)
        }
        self.level_proj = nn.ModuleDict()
        for level, target_dim in dim_map.items():
            if target_dim != base_dim:
                # 需要投影：Linear(base_dim → target_dim)
                self.level_proj[level] = nn.Linear(base_dim, target_dim)
            else:
                # 维度相同，跳过投影（恒等映射）
                self.level_proj[level] = None

    def forward(self, ref_img: torch.Tensor):
        """
        Forward pass — 从参考图像提取多尺度语义特征.

        Args:
            ref_img: (B, 3, H, W) 参考图像，值域 [-1, 1] 或 [0, 1]
                     会被 resize 到 224×224 以匹配 DINOv2 输入

        Returns:
            dict: {
                "e1":     (B, 32, unet_dim),       ← 注入 down_blocks[0]
                "e2":     (B, 16, unet_dim×2),     ← 注入 down_blocks[1]
                "e3":     (B, 8,  unet_dim×4),     ← 注入 down_blocks[2]
                "latent": (B, 4,  unet_dim×8),     ← 注入 mid_block
            }
            每个 value 可直接与 UNet 对应层特征做 add/concat 注入。

        注: last_hidden_state[:, 1:, :] 去掉 CLS Token，只保留 Patch Token
            Note: we discard the CLS token (index 0) and keep only patch tokens (indices 1..257)
        """
        # ── Step 1: Resize to DINOv2 fixed input size ──
        # DINOv2 要求固定 224×224 输入，patch_size=14 → 256 patches
        if ref_img.shape[-2:] != (224, 224):
            ref_small = F.interpolate(ref_img, size=(224, 224), mode="bilinear")
        else:
            ref_small = ref_img

        # ── Step 2: DINOv2 特征提取 ──
        # 根据 freeze_dinov2 控制是否追踪梯度
        # freeze=True:  torch.no_grad() → 纯推理，省显存
        # freeze=False: torch.enable_grad() → 允许微调 DINOv2
        ctx = torch.no_grad() if self.freeze_dinov2 else torch.enable_grad()
        with ctx:
            outputs = self.dinov2(ref_small)
            # last_hidden_state: (B, 257, dino_dim)
            #   [:, 0, :] = CLS token (丢弃)
            #   [:, 1:, :] = 256 Patch tokens (保留)
            features = outputs.last_hidden_state[:, 1:, :]  # (B, 256, dino_dim)

        # ── Step 3: 投影到基础维度 ──
        # DINOv2 dim → base_dim (可训练投影层)
        features = self.proj(features)  # (B, 256, base_dim)

        # ── Step 4: RWKV Semantic Pyramid ──
        # 256 → 32 → 16 → 8 → 4 tokens
        base_pyramid = self.semantic_pyramid(features)  # dict of (B, N, base_dim)

        # ── Step 5: 投影到 UNet 各层维度 ──
        # 每层从 base_dim 投影到对应 UNet 层的实际通道数
        output_pyramid = {}
        for level in ["e1", "e2", "e3", "latent"]:
            tokens = base_pyramid[level]  # (B, N, base_dim)
            proj = self.level_proj[level]
            output_pyramid[level] = (
                proj(tokens) if proj is not None else tokens  # 维度匹配时跳过投影
            )

        return output_pyramid


# ─────────────────────────────────────────────────────────────
# 测试代码 (Test / Sanity Check)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    t0 = time.time()
    print("Loading DINOv2...")

    model = (
        GlobalSemanticModule(
            dinov2_model_name="facebook/dinov2-base",
            base_dim=64,
            unet_dim=64,
        )
        .to(device)
        .eval()
    )

    print(f"DINOv2 loaded in {time.time() - t0:.1f}s")

    dummy_ref = torch.randn(2, 3, 480, 480).to(device)
    print(f"Running forward...")

    t0 = time.time()
    with torch.no_grad():
        output = model(dummy_ref)
    print(f"Forward done in {time.time() - t0:.1f}s")

    for k, v in output.items():
        print(f"{k}: {v.shape}")
