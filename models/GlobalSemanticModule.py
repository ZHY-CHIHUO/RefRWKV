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


# Semantic Pyramid
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


# Global Semantic Module
class GlobalSemanticModule(nn.Module):
    """
    全局语义提取器
    DINOv2 → 投影 → RWKV Semantic Pyramid
    """

    def __init__(
        self,
        dinov2_model_name: str = "facebook/dinov2-base",
        target_dim: int = 64,
        hidden_rate: int = 4,
        freeze_dinov2: bool = True,
    ):
        super().__init__()
        # 1. DINOv2 Backbone
        self.dinov2 = AutoModel.from_pretrained(dinov2_model_name)
        if freeze_dinov2:
            for param in self.dinov2.parameters():
                param.requires_grad = False

        # 2. 投影
        self.proj = nn.Linear(768, target_dim)

        # 3. RWKV Semantic Pyramid
        self.semantic_pyramid = RWKV_SemanticPyramid(
            dim=target_dim, hidden_rate=hidden_rate
        )

    def forward(self, ref_img: torch.Tensor):
        # Resize
        if ref_img.shape[-2:] != (224, 224):
            ref_small = F.interpolate(ref_img, size=(224, 224), mode="bilinear")
        else:
            ref_small = ref_img

        # DINO Features
        with torch.no_grad():
            outputs = self.dinov2(ref_small)
            features = outputs.last_hidden_state[:, 1:, :]  # remove cls token

        # 投影
        features = self.proj(features)

        # RWKV Semantic Pyramid
        sem_pyramid = self.semantic_pyramid(features)

        return sem_pyramid


# 测试
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GlobalSemanticModule(target_dim=64).to(device).eval()
    dummy_ref = torch.randn(2, 3, 480, 480).to(device)
    with torch.no_grad():
        sem_pyramid = model(dummy_ref)
    print({k: v.shape for k, v in sem_pyramid.items()})
    # 输出 e1:(2,32,64), e2:(2,16,64), e3:(2,8,64), latent:(2,4,64)
