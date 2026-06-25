"""
sd2_ref_adapter.py — RWKV Ref Adapter (轻量级 Ref 特征注入适配器)

将 RefDiffRWKV ref 管线输出的 rf1/rf2/rf3 特征，
通过逐层 1×1 卷积映射到 SD2 UNet 对应层通道数，
再以加法方式注入 SD2 UNet 的 down_blocks。

设计原则 (Design Principles):
    - 极轻量: 仅 3 层 1×1 Conv，总计 ~3M 参数
    - 零初始化: 训练初期 ref 贡献为零，逐步释放 ref 信息
    - 无上/下采样: 假设 rf 特征图尺寸已与 UNet 各层对齐

通道映射 (Channel Mapping):
    rf1 (384ch) → sd2_down0 ( 320ch)
    rf2 (768ch) → sd2_down1 ( 640ch)
    rf3 (1536ch)→ sd2_down2 (1280ch)

参考文献:
    T2I-Adapter: Mou et al., "T2I-Adapter: Learning Adapters to Dig out
                  More Controllable Ability for Text-to-Image Diffusion Models", 2023
    CRefDiff:    \Hou et al., "CRefDiff: ...", 2024

用法 (Usage):
    adapter = RWKV_Ref_Adapter(ref_dims=(384, 768, 1536), sd2_dims=(320, 640, 1280))
    rf1, rf2, rf3 = ref_diff_rwkv.extract_ref_features(LR, Ref)
    a1, a2, a3 = adapter(rf1, rf2, rf3)
    # 在 SD2 UNet 内部:
    # h = down_blocks[0](h) + a1
    # h = down_blocks[1](h) + a2
    # h = down_blocks[2](h) + a3
"""

import torch
import torch.nn as nn


class RWKV_Ref_Adapter(nn.Module):
    """
    RWKV Ref Adapter — 将 RefDiffRWKV 的 rf 特征注入 SD2 UNet.

    结构 (Structure):
        ┌───────────────────────────────────────────┐
        │ rf1 (B, 384, H,   W)                      │
        │   └─ Conv1×1(384 → 320, zero_init) ──► a1  │
        │                                           │
        │ rf2 (B, 768, H/2, W/2)                    │
        │   └─ Conv1×1(768 → 640, zero_init) ──► a2  │
        │                                           │
        │ rf3 (B, 1536, H/4, W/4)                   │
        │   └─ Conv1×1(1536 → 1280, zero_init) ──► a3 │
        └───────────────────────────────────────────┘

    参数 (Parameters):
        ref_dims: RefDiffRWKV 输出的 rf1/rf2/rf3 通道数，默认 (384, 768, 1536)
        sd2_dims: SD2 UNet down_blocks[0/1/2] 的通道数，默认 (320, 640, 1280)

    零初始化策略 (Zero-Init Strategy):
        所有权重初始化为 0，偏置初始化为 0。
        训练第 0 步时 adapter 输出全零 → ref 不干扰 SD2 UNet 原有生成能力。
        随着训练进行，权重逐渐学到非零值 → ref 信息逐步注入。
        这是 T2I-Adapter / ControlNet 等工作的标准做法。
    """

    def __init__(
        self,
        ref_dims: tuple = (384, 768, 1536),
        sd2_dims: tuple = (320, 640, 1280),
    ):
        super().__init__()

        ref_dim1, ref_dim2, ref_dim3 = ref_dims
        sd2_dim1, sd2_dim2, sd2_dim3 = sd2_dims

        # ── 三层 1×1 卷积（逐点通道映射）──
        # 1×1 Conv is equivalent to a per-pixel linear layer.
        # It maps channels without changing spatial resolution.
        # 1×1 卷积等价于逐像素线性变换，只改变通道数，不改变空间尺寸。

        self.conv1 = nn.Conv2d(
            ref_dim1, sd2_dim1,   # 384 → 320
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,            # 保留 bias 以便零初始化时输出完全为零
        )

        self.conv2 = nn.Conv2d(
            ref_dim2, sd2_dim2,   # 768 → 640
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.conv3 = nn.Conv2d(
            ref_dim3, sd2_dim3,   # 1536 → 1280
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # ── 零初始化 ──
        # Zero-initialize all weights and biases.
        # At step 0: adapter(x) = 0 for all x → ref has no effect.
        # Gradients will push weights away from zero during training.
        self._zero_init()

    def _zero_init(self):
        """
        Initialize all Conv weights and biases to zero.

        零初始化确保:
        - 训练初期 adapter 输出 ≡ 0
        - SD2 UNet 行为等同于无 ref 注入的原始模型
        - ref 信号随训练逐步"生长"进来
        """
        for conv in [self.conv1, self.conv2, self.conv3]:
            nn.init.constant_(conv.weight, 0.0)
            nn.init.constant_(conv.bias, 0.0)

    def forward(
        self,
        rf1: torch.Tensor,   # (B, 384, H,   W)   — RefDiffRWKV 第一层输出
        rf2: torch.Tensor,   # (B, 768, H/2, W/2) — RefDiffRWKV 第二层输出
        rf3: torch.Tensor,   # (B, 1536, H/4, W/4)— RefDiffRWKV 第三层输出
    ):
        """
        Forward pass — 逐层通道映射.

        Args:
            rf1: (B, 384, H,   W)    来自 RefMultiScaleProcessor 的 f1
            rf2: (B, 768, H/2, W/2)  来自 RefMultiScaleProcessor 的 f2
            rf3: (B, 1536, H/4, W/4) 来自 RefMultiScaleProcessor 的 f3

        Returns:
            a1: (B, 320, H,   W)     → 注入 SD2 down_blocks[0]
            a2: (B, 640, H/2, W/2)   → 注入 SD2 down_blocks[1]
            a3: (B, 1280, H/4, W/4)  → 注入 SD2 down_blocks[2]

        注 (Note):
            输出特征图分辨率与输入一致，不做上/下采样。
            空间对齐假设：RefDiffRWKV 的 patch_h/patch_w 与 SD2 UNet
            对应层的特征图分辨率已经匹配（均为 H/8 的 latent 空间）。
        """
        a1 = self.conv1(rf1)   # (B, 384, H,   W)  → (B, 320, H,   W)
        a2 = self.conv2(rf2)   # (B, 768, H/2, W/2)→ (B, 640, H/2, W/2)
        a3 = self.conv3(rf3)   # (B, 1536,H/4, W/4)→ (B, 1280,H/4, W/4)

        return a1, a2, a3

    def get_parameter_count(self) -> dict:
        """
        Count trainable parameters per layer (for logging).
        统计各层可训练参数量（用于日志输出）。

        Returns:
            dict: {"conv1": N1, "conv2": N2, "conv3": N3, "total": N_total}
        """
        counts = {}
        total = 0
        for name in ["conv1", "conv2", "conv3"]:
            conv = getattr(self, name)
            n = sum(p.numel() for p in conv.parameters() if p.requires_grad)
            counts[name] = n
            total += n
        counts["total"] = total
        return counts


# ─────────────────────────────────────────────────────────────
# 测试代码 (Test / Sanity Check)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 构造适配器
    adapter = RWKV_Ref_Adapter(
        ref_dims=(384, 768, 1536),
        sd2_dims=(320, 640, 1280),
    ).to(device)

    # 打印参数量
    param_counts = adapter.get_parameter_count()
    print("=" * 60)
    print("RWKV_Ref_Adapter — Parameter Count (参数量)")
    print("-" * 60)
    for k, v in param_counts.items():
        print(f"  {k}: {v:,}")
    print("-" * 60)
    print(f"  Total: ~{param_counts['total'] / 1e6:.1f}M parameters")
    print("=" * 60)

    # 模拟 RefDiffRWKV 输出的 rf 特征
    # 假设 latent 空间分辨率为 60×60 (对应 480×480 输入, patch_size=4 → 120, VAE 8× down → 60)
    B, H, W = 2, 60, 60  # batch=2, latent resolution=60×60

    rf1 = torch.randn(B, 384, H,   W).to(device)   # (2, 384, 60, 60)
    rf2 = torch.randn(B, 768, H//2, W//2).to(device)  # (2, 768, 30, 30)
    rf3 = torch.randn(B, 1536, H//4, W//4).to(device) # (2, 1536, 15, 15)

    # 前向推理
    with torch.no_grad():
        a1, a2, a3 = adapter(rf1, rf2, rf3)

    print(f"\nInput shapes:")
    print(f"  rf1: {rf1.shape}")
    print(f"  rf2: {rf2.shape}")
    print(f"  rf3: {rf3.shape}")
    print(f"\nOutput shapes (after zero-init, all values should be 0):")
    print(f"  a1:  {a1.shape}  | sum = {a1.sum().item():.6f}")
    print(f"  a2:  {a2.shape}  | sum = {a2.sum().item():.6f}")
    print(f"  a3:  {a3.shape}  | sum = {a3.sum().item():.6f}")

    # 验证零初始化
    assert a1.sum().item() == 0.0, "Zero-init failed: conv1 output is not all zeros"
    assert a2.sum().item() == 0.0, "Zero-init failed: conv2 output is not all zeros"
    assert a3.sum().item() == 0.0, "Zero-init failed: conv3 output is not all zeros"
    print("\n✓ Zero-initialization verified — all outputs are exactly zero.")
