# RefSRWKV: 基于 RWKV 骨干的参考引导超分辨率网络

**RefSRWKV** 是一个专为高分辨率图像（如遥感影像）设计的参考引导超分辨率（Reference-based Super-Resolution）先验网络。它结合了 RWKV 的线性复杂度序列建模能力与 U-Net 的多尺度特征提取优势，支持**任意倍率（Scale-Agnostic）** 的超分任务。

---

## 🌟 核心设计哲学

### 1. 固定内部分辨率 (Fixed Internal Resolution)

传统超分网络直接在输入分辨率上运行，导致大倍率（如 10x）时深层特征图过小（如 6×6），网络无法学习全局结构；而在原图分辨率（如 512×512）上运行又会导致显存爆炸。
RefSRWKV 采用**固定内部分辨率**架构：

- 无论物理输入的 LR 和 Ref 尺寸如何，网络内部**始终在固定的 `internal_size` 上进行特征提取**。
- **黄金法则**：通过 `PixelUnshuffle(4)` 将参考图无损折叠到通道维度，网络自动推导 `internal_size = HR_size // 4`。这不仅完美对齐了 LR 和 Ref 的特征尺寸，还保证了 $3 \times 4^2 = 48$ 通道与网络基础维度 `dim=48` 的严丝合缝。

### 2. 遥感特定的窗口化 RWKV (Windowed Bi-WKV)

- **8×8 窗口注意力**：将全局序列切分为 8×8 的局部窗口，大幅降低计算复杂度。
- **拒绝 `torch.roll`**：在自然图像（如 Swin Transformer）中，循环移位（Roll）是标配。但在**遥感图像**中，Roll 会将图像左边缘的“海洋”与右边缘的“陆地”强行拼接，产生严重的**地理伪影**。RefSRWKV 采用 **Zero-Padding** 替代 Roll，牺牲微小的跨边界交互，换取地理特征的绝对纯洁性。
- **P2 通道分段初始化**：为适配底层 CUDA WKV 算子 `C >= 16` 的硬性限制，在完整通道维度上进行分段赋值（`-0.5 * (g + 1)`），在数学上等效于分组衰减，同时避免了 Python 层面的循环调用开销。

---

## 🧱 关键模块解析

### `OmniShift` (多尺度动态卷积)

在训练阶段，并行计算 1×1、3×3、5×5 深度可分离卷积，并通过可学习的 `alpha` 和残差门控 `gate` 进行动态融合。
**推理优化**：在 `eval()` 模式下，通过 `reparam_5x5()` 将所有分支**精确重参数化为单个 5×5 卷积**，实现零额外推理开销。

### `GatedFusion` (置信度门控融合)

用于将参考图特征（Ref）注入低分辨率特征（LR）。

- **余弦相似度置信度**：计算 LR 与 Ref 特征的通道级余弦相似度。
- **线性映射修复**：摒弃了容易导致梯度截断的 `sigmoid(sim * 2.0)`，采用 `(sim + 1.0) / 2.0` 将相似度 `[-1, 1]` 线性映射到 `[0, 1]`，确保即使参考图不可靠，也不会产生极端的特征抹杀。

### `VRWKV_SpatialMix` (空间混合)

- **行列交替扫描**：在 8×8 窗口内，第一轮进行行优先扫描（H→W），第二轮将特征转置后进行列优先扫描（W→H），实现窗口内的完整二维上下文聚合。
- **P3 SE 通道门控**：在 WKV 输出后，接入 Squeeze-and-Excitation (SE) 模块，进行通道级的注意力重标定。

---

## ⚙️ 训练工程优化

### 1. 学习率调度 (SequentialLR)

彻底解决了手动修改 `pg["lr"]` 导致的 Warmup 与 CosineAnnealing 冲突问题。使用 PyTorch 原生的 `SequentialLR` 将 `LinearLR` (Warmup) 和 `CosineAnnealingLR` 无缝拼接，确保学习率曲线严格符合预期。

### 2. EMA 验证期隔离

在 Lightning 中，若 `val_check_interval < 1.0`（如每半个 epoch 验证一次），传统的 `on_validation_epoch_start` 会导致 EMA 权重污染后续的训练梯度。
**修复**：改用模型级别的钩子 `on_validation_model_eval` 和 `on_validation_model_train`，确保 EMA 权重**仅在验证步骤内部**临时生效。

### 3. Reference Dropout (盲超分鲁棒性)

在训练时，以 `ref_drop_prob`（推荐 0.1~0.2）的概率将参考图替换为 Batch 内的无关图像。这迫使 `GatedFusion` 学会关闭门控，使网络退化为纯单图超分路径，从而**强制主干网络学习不依赖参考图的底层恢复能力**，防止对参考图过度拟合。

### 4. 频域感知损失 (FFT Loss)

除了常规的 L1 Loss，引入 `torch.fft.rfft2` 计算频域 L1 损失。这能有效惩罚高频信息的丢失，显著提升遥感图像中**边缘锐度**和**纹理细节**的重建质量。

---

## 📝 配置与使用指南

得益于自动推导逻辑，YAML 配置变得极其简洁。您只需指定数据的物理裁剪尺寸（`patch_size`），网络会自动计算内部运行尺寸。

### 配置文件示例 (`configs/sr_prior_4.yaml`)

```yaml
model:
  inp_channels: 3
  out_channels: 3
  dim: 48
  num_blocks: [4, 6, 6, 8]
  num_refinement_blocks: 4
  scale: 4 # 超分倍率
  drop_path_rate: 0.1
  hidden_rate: 4
  learning_rate: 1.0e-4
  warmup_steps: 2000
  grad_clip_norm: 1.0
  ema_decay: 0.999
  use_ema: true
  ssim_weight: 0.0
  fft_weight: 0.2 # 频域损失权重
  ref_drop_prob: 0.1 # 参考图 Dropout 概率

data:
  root: "RefSR_data/HRMS_SCD"
  patch_size: 512 # ★ HR 裁剪尺寸 (网络自动推导 internal_size = 128)
  scale: 4
  batch_size: 8
  augment: true
  augment_ref: false

train:
  devices: 1
  precision: bf16-mixed
  max_epochs: 100
  val_check_interval: 0.5
  early_stopping_patience: 15
```

### 启动与热启动 (Hot-Start)

由于架构升级（修复了 CUDA 报错、优化了输出头），新旧模型在极少数参数（如 `spatial_decay`）上存在形状差异。训练脚本内置的 `load_weights_filtered` 会**自动跳过不匹配的参数，复用 95% 以上的核心权重**。

```bash
# 清理旧缓存（防止加载旧版 .pyc）
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# 热启动训练（使用 --load_weights 而非 --resume）
python scripts/train_sr_prior.py \
  --config configs/sr_prior_4.yaml \
  --load_weights checkpoints/old_best_model.ckpt
```

---

## ⚠️ 已知限制与注意事项

1. **输出激活函数**：模型最后使用 `torch.tanh()`。这是因为数据集在加载时统一归一化到了 `[-1, 1]` 值域。若您的数据集值域为 `[0, 1]`，请将 `tanh` 替换为 `sigmoid` 或直接输出并 `clamp(0, 1)`。
2. **CUDA 依赖**：空间 RWKV 路径强依赖自定义的 `bi_wkv` CUDA 算子。算子内部强制使用 `float32` 累加器以防止 `exp()` 溢出，这是保证训练不出现 NaN 的数学底线，不支持纯 CPU 推理。
3. **显存占用**：在 `patch_size=512`、`batch_size=8` 的配置下，单卡 24GB 显存（如 RTX 3090/4090）可流畅运行。若遇 OOM，请开启 `accumulate_grad_batches: 2` 并减半 `batch_size`。
