#!/usr/bin/env python
"""
SD2 UNet + LoRA 插入模块
=======================
将 LoRA 适配器注入 UNet2DConditionModel 的 attention 投影层和关键卷积层。

策略（按优先级）：
  1. PEFT 库自动匹配 → 最简洁，HYPIR 官方做法
  2. 手写 LoRA wrapper → 备选，完全可控

用法:
    from models.sd2_unet_lora import get_sd2_unet_lora
    unet = get_sd2_unet_lora(lora_rank=64)
    # unet 可直接用于训练/推理，基础权重自动冻结
"""

import math
import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel


# ============================================================
# 手写 LoRA（备选方案）
# ============================================================
class LoRALinear(nn.Module):
    """
    对 nn.Linear 注入 LoRA 低秩适配。
    原层权重冻结，新增 lora_A/lora_B 可训练。
    """
    def __init__(self, linear: nn.Linear, rank: int, alpha: float = None):
        super().__init__()
        self.linear = linear          # frozen
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank

        self.lora_A = nn.Parameter(torch.zeros(linear.in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, linear.out_features))

        # 初始化：A kaiming，B 全零 → 训练初 LoRA 贡献为零
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # 冻结原有权重
        for p in self.linear.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        base = self.linear(x)                              # frozen forward
        lora = (x @ self.lora_A) @ self.lora_B             # low-rank path
        return base + (self.alpha / self.rank) * lora


class LoRAConv2d(nn.Module):
    """
    对 nn.Conv2d 注入 LoRA。
    原卷积冻结，新增 1×1 conv_A + 1×1 conv_B。
    """
    def __init__(self, conv: nn.Conv2d, rank: int, alpha: float = None):
        super().__init__()
        self.conv = conv
        self.rank = rank
        self.alpha = alpha if alpha is not None else rank

        self.lora_A = nn.Conv2d(
            conv.in_channels, rank, kernel_size=1, bias=False
        )
        self.lora_B = nn.Conv2d(
            rank, conv.out_channels, kernel_size=1, bias=False
        )

        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

        for p in self.conv.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        base = self.conv(x)                                # frozen forward
        lora = self.lora_B(self.lora_A(x))                 # low-rank path
        return base + (self.alpha / self.rank) * lora


# ============================================================
# PEFT 方式（优先）
# ============================================================
def _apply_peft_lora(unet: UNet2DConditionModel, rank: int, alpha: float):
    """使用 peft 库插入 LoRA。"""
    from peft import LoraConfig, get_peft_model

    # 目标模块：注意力投影 + 首尾卷积（HYPIR 推荐，参数适中）
    target_modules = [
        "to_k", "to_q", "to_v", "to_out.0",   # Self/Cross Attention
        "conv_in", "conv_out",                 # 首尾卷积
    ]
    # 如需扩展至 ResNet 卷积，可取消注释以下行（参数会大幅增加）
    # target_modules += ["conv1", "conv2", "conv_shortcut", "conv"]

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
    )

    unet = get_peft_model(unet, lora_config)

    # 显式冻结所有非 LoRA 参数，确保基础模型不会被误训练
    for name, param in unet.named_parameters():
        if 'lora' not in name:
            param.requires_grad = False

    # 打印可训练参数统计
    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet.parameters())
    print(f"✅ PEFT LoRA 插入成功")
    print(f"   可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return unet


# ============================================================
# 手写方式（备选，与 PEFT 目标对齐）
# ============================================================
def _needs_lora_linear(name: str) -> bool:
    """判断某个 nn.Linear 是否应该插入 LoRA。"""
    targets = {"to_k", "to_q", "to_v", "to_out.0"}
    return any(name.endswith(t) for t in targets)


def _needs_lora_conv2d(name: str) -> bool:
    """判断某个 nn.Conv2d 是否应该插入 LoRA。"""
    # 只对首尾卷积注入，与 PEFT 设置保持一致
    targets = {"conv_in", "conv_out"}
    return any(name.endswith(t) for t in targets)


def _apply_manual_lora(unet: UNet2DConditionModel, rank: int, alpha: float):
    """手写 LoRA 注入，逐模块替换。"""
    replace_count = 0

    def _replace(parent, child_name, child):
        nonlocal replace_count
        if isinstance(child, nn.Linear) and _needs_lora_linear(child_name):
            setattr(parent, child_name, LoRALinear(child, rank, alpha))
            replace_count += 1
        elif isinstance(child, nn.Conv2d) and _needs_lora_conv2d(child_name):
            setattr(parent, child_name, LoRAConv2d(child, rank, alpha))
            replace_count += 1

    # 递归遍历所有子模块
    def _walk(module: nn.Module, parent=None):
        for name, child in list(module.named_children()):
            _replace(module, name, child)
            _walk(child, module)

    _walk(unet)

    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet.parameters())
    print(f"✅ 手写 LoRA 插入成功 ({replace_count} 个模块)")
    print(f"   可训练参数: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return unet


# ============================================================
# 主入口
# ============================================================
def get_sd2_unet_lora(
    sd2_model_id: str = "sd2-community/stable-diffusion-2-1-base",
    lora_rank: int = 64,
    lora_alpha: float = None,
    use_peft: bool = True,
    device: str = None,
    torch_dtype: torch.dtype = torch.float32,
) -> UNet2DConditionModel:
    """
    加载 SD2 UNet 并插入 LoRA。

    Args:
        sd2_model_id:  HuggingFace 模型 ID
        lora_rank:     LoRA 秩（默认 64，HYPIR 消融实验推荐值）
        lora_alpha:    LoRA scaling 系数（默认 = rank）
        use_peft:      优先尝试 peft 库
        device:        目标 device
        torch_dtype:   模型精度（float32 或 bfloat16）

    Returns:
        插入了 LoRA 的 UNet2DConditionModel（基础权重冻结）
    """
    if lora_alpha is None:
        lora_alpha = lora_rank

    print(f"加载 SD2 UNet: {sd2_model_id}")
    unet = UNet2DConditionModel.from_pretrained(
        sd2_model_id,
        subfolder="unet",
        torch_dtype=torch_dtype,
    )

    # --- 尝试 PEFT ---
    if use_peft:
        try:
            import peft
            unet = _apply_peft_lora(unet, lora_rank, lora_alpha)
        except Exception as e:
            print(f"⚠️  PEFT 插入失败: {e}")
            print("   回退到手写 LoRA wrapper...")
            unet = _apply_manual_lora(unet, lora_rank, lora_alpha)
    else:
        unet = _apply_manual_lora(unet, lora_rank, lora_alpha)

    if device is not None:
        unet = unet.to(device)

    return unet


# ============================================================
# 快速测试（直接运行本文件时触发）
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("SD2 UNet + LoRA 测试")
    print("=" * 60)

    # 1. 插入 LoRA
    unet = get_sd2_unet_lora(lora_rank=64, device="cuda")

    # 2. 构造虚拟 latent 做一次前向传播
    #    SD2: latent shape = (B, 4, H/8, W/8)，以 480×480 输入为例
    B, C, H_lat, W_lat = 1, 4, 60, 60
    sample = torch.randn(B, C, H_lat, W_lat, device="cuda")
    # timestep 必须为 float 类型
    timestep = torch.tensor([500.0], device="cuda")
    #    空 prompt 对应的 encoder_hidden_states (SD2 用 1024 维)
    encoder_hidden_states = torch.zeros(B, 77, 1024, device="cuda")

    with torch.no_grad():
        out = unet(sample, timestep, encoder_hidden_states).sample

    print(f"输入 sample  shape: {sample.shape}")
    print(f"输出 output  shape: {out.shape}")
    print(f"前向传播: ✅ 通过")
    print(f"模型参数量: {sum(p.numel() for p in unet.parameters()):,}")
    print(f"可训练参数: {sum(p.numel() for p in unet.parameters() if p.requires_grad):,}")