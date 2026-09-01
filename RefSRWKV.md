# RefSRWKV

RefSRWKV 是面向 RGB 图像的参考引导超分辨率先验网络。模型以窗口化双向 RWKV 为骨干，结合多尺度参考特征融合与残差重建头；数据集输入和输出的数值范围均为 `[-1, 1]`。

## 输入与尺寸约定

```text
lr:   [B, C_in, H_lr, W_lr]
ref:  [B, C_in, H_ref, W_ref]
out:  [B, C_out, H_ref, W_ref]
```

- `ref_channels` 必须等于 `inp_channels`，因为参考图颜色统计以 LR 图像为目标进行对齐。
- `dim` 必须是 16 的倍数，以满足 CUDA Bi-WKV 算子的通道约束。
- `hr_size` 是训练 HR patch 的边长，用于定义模型内部网格，必须能被 32 整除。
- `scale` 定义数据集裁剪时的 LR/HR 尺寸关系；网络内部始终使用固定网格，因此可用于任意整数倍率的数据。

内部特征尺寸为 `internal_size = hr_size / 4`。参考图先经过 `PixelUnshuffle(4)` 折叠到 HR/4 网格，再经过三级下采样；解码器回到 HR/4 网格后使用两级 `PixelShuffle(2)` 生成高分辨率残差。最终输出为：

```text
clamp(bicubic(lr) + residual, -1, 1)
```

残差输出卷积采用零初始化，训练开始时的预测即为双三次插值基线。

## 空间混合与参考融合

`VRWKV_SpatialMix` 将特征分成 8x8 窗口，并在窗口内执行行优先与列优先的 Bi-WKV 扫描。移位窗口在上侧和左侧进行零填充，在下侧和右侧补齐窗口尺寸，因此图像边界不会发生循环拼接。

`GatedFusion` 在四个尺度融合 LR 与参考特征。它以余弦相似度作为置信度，将其映射到 `[0, 1]`，并通过可学习门控注入融合特征：

```text
lr_feature + gate(fused) * confidence * fused
```

训练时可通过 `ref_drop_prob` 随机替换参考图，以增强无可靠参考场景下的恢复能力。批量大小为 1 时，替换图像为双三次上采样的 LR 图像。

## CUDA 环境

空间 RWKV 路径依赖 `models/cuda/bi_wkv.cpp` 与 `models/cuda/bi_wkv_kernel.cu`。首次使用时，PyTorch 会通过 `torch.utils.cpp_extension.load` 编译并加载扩展。

建议使用带 CUDA 的 PyTorch 环境。项目已在 Conda 环境 `rwkv7`、PyTorch `2.10.0+cu128`、CUDA `12.8` 和 RTX 5060 Ti 上完成训练路径验证。

所需 Python 依赖：

```text
torch
einops
pytorch-lightning
torchvision
Pillow
PyYAML
lmdb
pyiqa  # 可选；不可用时自动使用手写 SSIM
```

## 数据集目录

PNG 数据集按同名文件配对：

```text
<data_root>/<split>/LR/*.png
<data_root>/<split>/HR/*.png
<data_root>/<split>/Ref/*.png
```

图像以 RGB 读取并归一化到 `[-1, 1]`。随机裁剪先采样整数 LR 坐标，再按 `data.scale` 映射到 HR 与参考图坐标，以保持空间对齐。

## 训练

在仓库根目录使用 `rwkv7` 环境启动：

```bash
conda run -n rwkv7 python scripts/train_sr_prior.py --config configs/sr_prior_4.yaml
conda run -n rwkv7 python scripts/train_sr_prior.py --config configs/sr_prior_10.yaml
```

| 配置 | HR patch | LR patch | 倍率 | 内部网格 | batch size |
| --- | ---: | ---: | ---: | ---: | ---: |
| `configs/sr_prior_4.yaml` | 512 | 128 | 4 | 128 | 4 |
| `configs/sr_prior_10.yaml` | 480 | 48 | 10 | 120 | 4 |

两个配置均使用 BF16 混合精度与梯度累积。RTX 5060 Ti（16 GiB）已完成上述配置的单次 batch=4 前向、反向与优化器更新。

训练模块由 L1 损失与可选 SSIM、FFT 损失组成；EMA 仅在真实 optimizer step 后更新，并在验证和测试期间临时应用。学习率默认使用 `ReduceLROnPlateau`，调度器在每次验证运行结束时读取本次聚合后的 `val_loss`，而不是只在 epoch 末读取一次。

### 学习率调度

两个独立训练配置都使用以下 plateau 规则（具体初始学习率、衰减因子和下限以对应 YAML 为准）：

| 参数 | 含义 |
| --- | --- |
| `lr_patience: 2` | 连续 2 次验证没有有效改善后，下一次无效验证触发衰减；因此第 3 次 bad validation 降 LR。 |
| `lr_threshold: 1.0e-4` | `mode=min` 下，`val_loss` 必须比历史最佳值下降 **超过** 此绝对阈值才算改善。设为 `0` 可关闭最小改善门槛。 |
| `lr_factor` | 触发时将当前学习率乘以该系数。 |
| `lr_min` | 学习率下限，不会继续降到该值以下。 |

`val_check_interval` 决定每个 epoch 的验证次数，`lr_patience` 也按这个验证次数计数。例如 `0.2` 通常表示每个 epoch 验证 5 次，因此 patience 不再代表 epoch 数。任何一次达到阈值的中间验证都会立即更新历史最佳值并重置 bad-validation 计数，不会被 epoch 末结果覆盖。`max_epochs` 只是安全上限，实际停止仍由 `early_stopping_patience` 控制。

### Checkpoint

`--resume` 用于结构和调度器配置均兼容的 Lightning checkpoint，会恢复模型、optimizer、plateau 计数和 EMA 状态。脚本只接受能够明确识别为“按验证计数”的 plateau 状态，避免把其他计数单位误当成验证计数。`--load_weights` 用于热启动：只加载同名且形状匹配的模型权重，optimizer、调度器和计数器从当前配置重新开始；不匹配参数保留当前初始化。模型结构变更、调度器语义变更或只想使用已有权重时，应使用 `--load_weights`。

```bash
conda run -n rwkv7 python scripts/train_sr_prior.py \
  --config configs/sr_prior_4.yaml \
  --load_weights /path/to/weights.ckpt
```

断点续训示例：

```bash
conda run -n rwkv7 python scripts/train_sr_prior.py \
  --config configs/sr_prior_4.yaml \
  --resume checkpoints/refrwkv_sr_4/last.ckpt
```

## 评测

`scripts/eval_four_settings.py` 评估 bicubic、无参考、真实参考和理想参考四种设置。`--hr_size` 必须与 checkpoint 训练时的 HR patch 一致：

```bash
conda run -n rwkv7 python scripts/eval_four_settings.py \
  --ckpt checkpoints/refrwkv_sr_4/last.ckpt \
  --scale 4 \
  --hr_size 512
```

10x checkpoint 使用 `--scale 10 --hr_size 480`。评测默认加载 EMA 权重，并在推理前调用 `prepare_for_inference()`。

## 扩散阶段集成

`scripts/train_sd2_gan.py` 构建 SR prior 时优先读取 `model.sr.hr_size`；未设置时使用 `data.patch_size`。这使 SR 内部网格与扩散训练 patch 保持一致。需要有意使用不同网格时，可在 `model.sr.hr_size` 中显式指定。
