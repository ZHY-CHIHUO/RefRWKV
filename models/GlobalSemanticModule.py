# models/GlobalSemantic.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import sys
from pathlib import Path

# 添加项目根目录到 sys.path
root_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, root_dir)
from models.RefSRWKV import RUN_CUDA


class RWKV_SemanticAggregator(nn.Module):
    """RWKV Semantic Aggregator (双向 RWKV)"""

    def __init__(self, dim: int, num_tokens: int = 32, hidden_rate: int = 4):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens

        # Query Token + Position
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
        q = self.query + self.query_pos
        q = q.expand(B, -1, -1)

        # 拼接 Query + Patch Token
        seq = torch.cat([q, ref_features], dim=1)
        seq = self.input_norm(seq)
        T = seq.shape[1]

        # 投影
        k = self.key(seq)
        v0 = self.value(seq)
        r = torch.sigmoid(self.receptance(seq))

        # 双向 RWKV
        v_forward = RUN_CUDA(self.decay / T, self.first / T, k, v0)
        k_rev = torch.flip(k, dims=[1])
        v_rev_input = torch.flip(v0, dims=[1])
        v_backward = RUN_CUDA(self.decay / T, self.first / T, k_rev, v_rev_input)
        v_backward = torch.flip(v_backward, dims=[1])
        v = 0.5 * (v_forward + v_backward)

        x = r * v
        x = self.output(x)

        # 提取 Query 部分 + Residual + FFN
        semantic_tokens = x[:, : self.num_tokens, :] + q
        semantic_tokens = semantic_tokens + self.ffn(self.norm(semantic_tokens))
        return semantic_tokens


# Semantic Pyramid (基础版本，输出固定的 dim)
class RWKV_SemanticPyramid(nn.Module):
    """32 → 16 → 8 → 4 Tokens 多尺度 Semantic Pyramid"""

    def __init__(self, dim: int, hidden_rate: int = 4):
        super().__init__()
        self.agg32 = RWKV_SemanticAggregator(
            dim, num_tokens=32, hidden_rate=hidden_rate
        )
        self.agg16 = RWKV_SemanticAggregator(
            dim, num_tokens=16, hidden_rate=hidden_rate
        )
        self.agg8 = RWKV_SemanticAggregator(dim, num_tokens=8, hidden_rate=hidden_rate)
        self.agg4 = RWKV_SemanticAggregator(dim, num_tokens=4, hidden_rate=hidden_rate)

    def forward(self, ref_features):
        sem32 = self.agg32(ref_features)
        sem16 = self.agg16(sem32)
        sem8 = self.agg8(sem16)
        sem4 = self.agg4(sem8)
        return {"e1": sem32, "e2": sem16, "e3": sem8, "latent": sem4}


# Global Semantic Module (支持多维度输出)
class GlobalSemanticModule(nn.Module):
    """
    全局语义提取器
    DINOv2 → 投影 → RWKV Semantic Pyramid → U-Net 层级维度投影

    参数:
        dinov2_model_name: DINOv2 模型名（自动读取 hidden_size）
        base_dim:          金字塔内部的工作维度
        unet_dim:          U-Net 第一层的 embed_dim（后续层自动 ×1, ×2, ×4, ×8）
        hidden_rate:       RWKV FFN 隐藏层倍率
        freeze_dinov2:     是否冻结 DINOv2
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

        # 1. DINOv2 Backbone（自动检测 hidden_size）
        self.dinov2 = AutoModel.from_pretrained(dinov2_model_name)
        dino_dim = self.dinov2.config.hidden_size  # 768 for base, 384 for small, ...
        if freeze_dinov2:
            for param in self.dinov2.parameters():
                param.requires_grad = False

        # 2. 投影到基础维度
        self.proj = nn.Linear(dino_dim, base_dim)  # ← 不再写死 768

        # 3. RWKV Semantic Pyramid
        self.semantic_pyramid = RWKV_SemanticPyramid(
            dim=base_dim, hidden_rate=hidden_rate
        )

        # 4. 投影到 U-Net 各层实际维度
        #    金字塔输出全是 (B, N, base_dim)
        #    → 分别投影到 enc1(×1), enc2(×2), enc3(×4), latent(×8) 的通道数
        dim_map = {
            "e1":     unet_dim,
            "e2":     unet_dim * 2,
            "e3":     unet_dim * 4,
            "latent": unet_dim * 8,
        }
        self.level_proj = nn.ModuleDict()
        for level, target_dim in dim_map.items():
            if target_dim != base_dim:
                self.level_proj[level] = nn.Linear(base_dim, target_dim)
            else:
                self.level_proj[level] = None  # 无需投影

    def forward(self, ref_img: torch.Tensor):
        # Resize 到 DINOv2 的固定输入尺寸
        if ref_img.shape[-2:] != (224, 224):
            ref_small = F.interpolate(ref_img, size=(224, 224), mode="bilinear")
        else:
            ref_small = ref_img

        # DINOv2 特征（根据 freeze 状态决定是否追踪梯度）
        ctx = torch.no_grad() if self.freeze_dinov2 else torch.enable_grad()
        with ctx:
            outputs = self.dinov2(ref_small)
            features = outputs.last_hidden_state[:, 1:, :]  # (B, 256, dino_dim)

        # 投影 → 金字塔
        features = self.proj(features)                       # (B, 256, base_dim)
        base_pyramid = self.semantic_pyramid(features)

        # 层级维度投影
        output_pyramid = {}
        for level in ["e1", "e2", "e3", "latent"]:
            tokens = base_pyramid[level]                     # (B, N, base_dim)
            proj = self.level_proj[level]
            output_pyramid[level] = proj(tokens) if proj is not None else tokens

        return output_pyramid


# 测试
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 指定目标维度
    target_dims = {"e1": 64, "e2": 128, "e3": 256, "latent": 512}
    model = GlobalSemanticModule(base_dim=64, target_dims=target_dims).to(device).eval()
    dummy_ref = torch.randn(2, 3, 480, 480).to(device)
    with torch.no_grad():
        output = model(dummy_ref)
    for k, v in output.items():
        print(f"{k}: {v.shape}")
