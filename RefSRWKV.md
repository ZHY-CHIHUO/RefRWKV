# RefSRWKV

RefSRWKV 是面向遥感图像超分辨率的 CUDA Bi-WKV U-Net。网络接收 LR 图像和位于 HR 网格的参考图，在四个特征尺度融合两路信息，并在 bicubic 上采样结果上预测 HR 残差。PNG 数据从加载、训练到评测始终使用 `[-1, 1]` 值域。

## 原生空间契约

```text
lr:   [B, C_in, H, W]
ref:  [B, C_in, H * scale, W * scale]
out:  [B, C_out, H * scale, W * scale]
```

`scale` 是模型唯一的空间倍率：

```text
Ref -> PixelUnshuffle(scale) -> LR 网格参考特征
LR  -> 原生 LR 网格 U-Net      -> LR 网格重建特征
LR 网格特征 -> 输出头           -> LR * scale 的 HR 残差
bicubic(LR, LR * scale) + 残差  -> 输出
```

前向会严格断言 `ref.shape[-2:] == lr.shape[-2:] * scale`，不会静默缩放尺寸错误的参考图。为通过三级 U-Net 下采样，LR 仅在右侧和下侧 replicate padding 到 8 的倍数；Ref 同步补齐 `scale` 倍像素。输出随后裁回原始 `LR * scale` 尺寸。因此训练 crop、任意全图和非 8 倍尺寸推理都遵守同一几何规则。

训练统一使用 `run.lr_patch: 48`，HR crop 自动为 `48 * scale`：

| 倍率 | 训练 LR | 训练 HR | Ref 折叠 | 默认输出头 |
| ---: | ---: | ---: | ---: | --- |
| x2 | 48 | 96 | x2 | progressive |
| x4 | 48 | 192 | x4 | progressive |
| x10 | 48 | 480 | x10 | direct |

验证和测试保留每张图片的原生全图尺寸。

## 归一化与参考融合

所有卷积式 LR/Ref 分支使用 `RMSNorm2d`：RMS 仅在每个像素的通道维度计算，不做空间归约，也不减均值。它保留 DC/颜色分量，同时避免 GroupNorm 因训练 crop 与测试全图尺寸不同而产生统计差异。RWKV Block 内保留 token 维度的 `LayerNorm`，同样不依赖图像大小。

`GatedFusion` 的门控由两个直接作用于 `H x W` 特征的 1x1 卷积组成，因此是逐像素门控，而非每张图一个 SE 标量：

```text
lr_feature + sigmoid(Conv1x1(ReLU(Conv1x1(fused)))) * confidence * fused
```

`model.color_match` 控制唯一可选的全图统计：

| 配置 | 行为 |
| --- | --- |
| `global` | 在参考编码前，将 Ref 各通道均值/标准差匹配到 bicubic LR |
| `none` | 保持 Ref 原始颜色，不进行匹配 |

尺寸泛化实验必须使用与 checkpoint 相同的设置。配对参考存在明显色彩偏移时优先 `global`；`none` 是检验全图颜色统计是否真正带来收益的干净消融。

## 窗口调度

每个 RWKV Block 只执行一次窗口内 Bi-WKV。`offsets` 按 block 索引选择窗口原点；`[0, 4]` 表示连续 block 依次使用正常、平移、正常、平移窗口，并不表示单个 block 内计算两次 WKV。

默认的 stage-local 调度在 U-Net 两侧保持对称：

| Stage | window / offsets |
| --- | --- |
| enc1、dec1、refine | 8 / `[0, 4]` |
| enc2、dec2 | 8 / `[0, 4]` |
| enc3、dec3 | 4 / `[0, 2]` |
| latent | 3 / `[0, 1]` |

LR48 的网格为 `48 -> 24 -> 12 -> 6`；latent 的 `3/[0,1]` 恰好划分 6x6 网格，无训练期 padding。任意兼容调度都可以写入 `model.windows`，例如统一的 `8/[0, 3, 6]` 适合受控消融。

## 输出头

`model.upsampler` 由 YAML 显式控制：

| 输出头 | 结构 | 适用倍率 |
| --- | --- | --- |
| `progressive` | 分解为多级 `Conv -> PixelShuffle -> ReLU` | x2、x3、x4；x4 为两级 x2 |
| `direct` | `Conv(dim, out_channels * scale^2) -> PixelShuffle(scale)` | x10 等高倍率 |

x4 的渐进式输出头会在中间 2x 网格保留特征表示。x10 的直出头将主要卷积保留在 LR 网格，只在最后一步展开空间相位，显著降低高分辨率激活开销。

## 配置布局

```text
configs/sr_prior_base.yaml       网络、优化器和窗口公共默认值
configs/datasets/*.yaml          数据集事实、路径、原始与准备后尺寸
configs/runs/*.yaml              倍率、LR crop、损失和 run 专属差异
```

每个 run 定义 `run.scale` 和 `run.lr_patch`。训练入口会把派生的 `data.train_lr_patch`、`data.train_hr_patch`、`data.scale`、窗口调度、输出头和架构签名写入 `train_config.yaml` 与 checkpoint。

当前提供的 run：

| Run | 训练 crop | 参考模式 | 验证 | 输出头 |
| --- | --- | --- | --- | --- |
| `aid_x4_l1` | LR48 / HR192，x4 | bicubic LR | AID 原生全图 | progressive |
| `ucmerced_x4` | LR48 / HR192，x4 | bicubic LR | 原生全图 | progressive |
| `hrms_scd_x4` | LR48 / HR192，x4 | 配对 Ref | 原生全图 | progressive |
| `real_refrssrd_x10` | LR48 / HR480，x10 | 配对 Ref | 原生全图 | direct |

## 训练

从头训练 AID x4：

```bash
conda run -n rwkv7 python scripts/train_sr_prior.py \
  --config configs/runs/aid_x4_l1.yaml
```

checkpoint 写入 `checkpoints/refrwkv_sr_aid_x4_l1/`，TensorBoard 写入 `logs/refrwkv_sr/aid_x4_l1/`。该 run 使用纯 L1 和 50,000 个 optimizer step。全图验证使用固定数量的图片；预处理后尺寸一致的数据集可提高验证 batch，混合尺寸数据集保持 batch 1。

`ReduceLROnPlateau`、checkpoint 选择和可选 early stopping 都使用聚合后的全图 `val_loss`。`lr_patience` 的单位是验证次数，`lr_threshold` 是最小绝对有效改善量。AID 默认每个 epoch 验证一次。

`--resume` 仅用于完全相同的原生 LR run，并恢复 optimizer、scheduler 与 EMA。切换数据集、倍率、输出头、crop 策略或窗口调度时使用 `--load_weights`；它只加载名称和形状均一致的张量，并重新初始化 optimizer、scheduler 与 EMA。

## 评测

在 AID 上评测 bicubic 和单图超分参考模式：

```bash
conda run -n rwkv7 python scripts/eval_four_settings.py \
  --ckpt checkpoints/refrwkv_sr_aid_x4_l1/last.ckpt \
  --data data/remote_sensing/prepared/AID \
  --splits test \
  --settings bicubic sisr_ref \
  --batch-size 1
```

运行尺寸泛化自检：

```bash
conda run -n rwkv7 python scripts/eval_size_generalization.py \
  --ckpt checkpoints/refrwkv_sr_aid_x4_l1/last.ckpt \
  --data data/remote_sensing/prepared/AID \
  --split test \
  --reference sisr \
  --lr-sizes 48 64 96 150 full
```

该脚本输出各尺寸的 PSNR/SSIM，并默认读取 checkpoint 的 `color_match`。用 `--color-match global` 或 `--color-match none` 可以在同一 checkpoint 上做颜色匹配消融。源图小于所请求 crop 时会标记 `N/A`，不会 resize 伪造结果。准备后的 AID 是 HR512/LR128，因此 LR150 需要至少 HR600 的数据，不能在该 AID split 上进行原生评测。

## CUDA 要求

空间 RWKV 只使用 `models/cuda/bi_wkv.cpp` 和 `models/cuda/bi_wkv_kernel.cu`。首次 CUDA 前向会编译并加载扩展；不提供 CPU 前向或 PyTorch 等价 WKV 实现。
