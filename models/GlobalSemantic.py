# models/GlobalSemantic.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from torch.utils.cpp_extension import load

wkv_cuda = load(
    name="bi_wkv",
    sources=["./cuda/bi_wkv.cpp", "./cuda/bi_wkv_kernel.cu"],
    verbose=True,
    extra_cuda_cflags=[
        "-res-usage",
        "--maxrregcount 60",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        "-gencode arch=compute_120,code=sm_120",
    ],
)


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, u, k, v):
        half_mode = w.dtype == torch.half
        bf_mode = w.dtype == torch.bfloat16
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = wkv_cuda.bi_wkv_forward(w, u, k, v)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        w, u, k, v = ctx.saved_tensors
        half_mode = w.dtype == torch.half
        bf_mode = w.dtype == torch.bfloat16
        gw, gu, gk, gv = wkv_cuda.bi_wkv_backward(
            w.float().contiguous(),
            u.float().contiguous(),
            k.float().contiguous(),
            v.float().contiguous(),
            gy.float().contiguous(),
        )
        if half_mode:
            return (gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            return (gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            return (gw, gu, gk, gv)


def RUN_CUDA(w, u, k, v):
    return WKV.apply(w.cuda(), u.cuda(), k.cuda(), v.cuda())

class RWKV_SemanticAggregator(nn.Module):
    """
    基于 RWKV 双向 WKV 的语义 Token 聚合器。
    通过拼接可学习的 Query 和参考特征，进行正向 + 反向 WKV 扫描，
    然后提取 Query 部分作为聚合后的语义 Token。
    """

    def __init__(self, dim: int, num_tokens: int = 32, hidden_rate: int = 4):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens

        # 可学习的语义 Token (作为 Query)
        self.query = nn.Parameter(torch.randn(1, num_tokens, dim))

        # RWKV 风格的投影
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)

        # 可学习的衰减和补偿参数 (形状: (1, dim))
        self.decay = nn.Parameter(torch.randn(1, dim) * 0.1)
        self.first = nn.Parameter(torch.randn(1, dim) * 0.1)

        # FFN 增强
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * hidden_rate),
            nn.GELU(),
            nn.Linear(dim * hidden_rate, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, ref_features: torch.Tensor) -> torch.Tensor:
        """
        ref_features: (B, M, dim)   M 为 patch 数（如 256）
        返回: (B, num_tokens, dim)  聚合后的语义 Token
        """
        B, M, C = ref_features.shape

        # 1. 扩展 Query
        q = self.query.expand(B, -1, -1)  # (B, num_tokens, dim)

        # 2. 拼接: [Query, Ref_features]
        seq = torch.cat([q, ref_features], dim=1)  # (B, num_tokens+M, dim)
        T = seq.size(1)

        # 3. RWKV 投影
        k = self.key(seq)  # (B, T, dim)
        v = self.value(seq)
        r = torch.sigmoid(self.receptance(seq))

        # 4. 正向 WKV 扫描
        # 注意：RUN_CUDA 期望的 decay/first 形状为 (1, dim)，这里除以 T
        v = RUN_CUDA(self.decay / T, self.first / T, k, v)

        # 5. 反向 WKV 扫描 (翻转序列再扫描)
        k_rev = torch.flip(k, dims=[1])
        v_rev = torch.flip(v, dims=[1])
        v_rev = RUN_CUDA(self.decay / T, self.first / T, k_rev, v_rev)
        v = torch.flip(v_rev, dims=[1])  # 恢复原始顺序

        # 6. 门控 + 输出投影
        x = r * v
        x = self.output(x)

        # 7. 提取 Query 部分 (前 num_tokens 个)
        aggregated = x[:, : self.num_tokens, :]

        # 8. 残差 + FFN
        aggregated = aggregated + self.ffn(self.norm(aggregated))

        return aggregated


class SemanticTokenAggregator(nn.Module):
    """
    语义 Token 聚合模块 (STA)
    论文中: DiNOv2 特征 → 投影 → 自注意力 + 交叉注意力 + FFN → 紧凑语义 Token
    """

    def __init__(self, dim: int, num_tokens: int = 32, num_heads: int = 8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, num_tokens, dim))
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, ref_features: torch.Tensor) -> torch.Tensor:
        """
        ref_features: (B, M, dim)   M 为 patch 数（如 256）
        返回: (B, num_tokens, dim)  聚合后的语义 Token
        """
        B = ref_features.size(0)
        q = self.query.expand(B, -1, -1)  # (B, num_tokens, dim)

        # Self-attention on queries
        q = self.norm1(q + self.self_attn(q, q, q)[0])
        # Cross-attention with reference features
        q = self.norm2(q + self.cross_attn(q, ref_features, ref_features)[0])
        # FFN
        q = self.norm3(q + self.ffn(q))
        return q


class GlobalSemanticModule(nn.Module):
    """
    全局语义提取器 (DiNOv2 + 投影 + 可选聚合器)
    支持 Transformer STA 或 RWKV 聚合器
    """

    def __init__(
        self,
        dinov2_model_name: str = "facebook/dinov2-base",
        target_dim: int = 64,
        num_tokens: int = 32,
        num_heads: int = 8,  # 仅 Transformer 使用
        hidden_rate: int = 4,  # 仅 RWKV 使用
        freeze_dinov2: bool = True,
        use_rwkv: bool = True,  # True: RWKV聚合, False: Transformer聚合
    ):
        super().__init__()
        # 1. 加载 DiNOv2 模型
        self.dinov2 = AutoModel.from_pretrained(dinov2_model_name)
        if freeze_dinov2:
            for param in self.dinov2.parameters():
                param.requires_grad = False

        # 2. 投影层: 768 -> target_dim
        self.proj = nn.Linear(768, target_dim)

        # 3. 选择聚合器
        self.use_rwkv = use_rwkv
        if use_rwkv:
            # 确保 RUN_CUDA 已导入（需要在文件顶部从 RefDiffRWKV 导入）
            from .RefDiffRWKV import RUN_CUDA

            self.aggregator = RWKV_SemanticAggregator(
                dim=target_dim,
                num_tokens=num_tokens,
                hidden_rate=hidden_rate,
            )
        else:
            self.aggregator = SemanticTokenAggregator(
                dim=target_dim,
                num_tokens=num_tokens,
                num_heads=num_heads,
            )

    def forward(self, ref_img: torch.Tensor) -> torch.Tensor:
        """
        ref_img: (B, 3, H, W)  任意分辨率，会内部 resize 到 224x224
        返回: (B, num_tokens, target_dim)  聚合后的语义 Token
        """
        # 1. 缩放参考图像到 224x224
        if ref_img.shape[-2:] != (224, 224):
            ref_small = F.interpolate(ref_img, size=(224, 224), mode="bilinear")
        else:
            ref_small = ref_img

        # 2. DiNOv2 提取特征
        with torch.no_grad():
            outputs = self.dinov2(ref_small)
            features = outputs.last_hidden_state  # (B, 257, 768)
            features = features[:, 1:, :]  # 移除 cls token → (B, 256, 768)

        # 3. 投影到目标维度
        features = self.proj(features)  # (B, 256, target_dim)

        # 4. 聚合
        sem_tokens = self.aggregator(features)  # (B, num_tokens, target_dim)

        return sem_tokens


if __name__ == "__main__":
    # 检查 CUDA 是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"测试设备: {device}")

    # 创建模型并移动到设备
    model = GlobalSemanticModule(target_dim=64, use_rwkv=True)
    model.to(device)
    model.eval()

    # 生成随机输入（注意：DiNOv2 需要输入在 CPU 或 CUDA 上都能运行）
    dummy_ref = torch.randn(2, 3, 480, 480).to(device)

    with torch.no_grad():
        tokens = model(dummy_ref)

    print(f"输出形状: {tokens.shape}")  # 预期 (2, 32, 64)
    print("测试通过！")
