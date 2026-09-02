# RefSRWKV

RefSRWKV 是面向 RGB 图像的参考引导超分辨率先验网络。模型以窗口化双向 RWKV 为骨干，结合多尺度特征融合与残差重建头；数据集输入和输出的数值范围均为 `[-1, 1]`。当前的单图超分（SISR）数据管线把 bicubic 上采样的 LR 作为 Ref，不依赖同类或外部语义参考图。

## 输入与尺寸约定

```text
lr:   [B, C_in, H_lr, W_lr]
ref:  [B, C_in, H_ref, W_ref]
out:  [B, C_out, H_ref, W_ref]
```

- `ref_channels` 必须等于 `inp_channels`，因为参考图颜色统计以 LR 上采样图为目标进行对齐。
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

训练时可通过 `ref_drop_prob` 随机替换参考图，以增强 SISR 场景下的恢复能力。每个样本独立决定是否替换，和 batch size 无关；替换值始终是该样本 LR 的双三次上采样。`configs/sr_prior_hrms_scd_x4.yaml`、`configs/sr_prior_ucmerced.yaml` 和 `configs/sr_prior_aid.yaml` 当前设置为 `ref_drop_prob: 0.2`。

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
pyiqa  # 可选；不可用时自动使用手写 SSIM
```

## 数据集目录

PNG 数据集按同名文件配对：

```text
<data_root>/<split>/LR/*.png
<data_root>/<split>/HR/*.png
<data_root>/<split>/Ref/*.png
```

图像以 RGB 读取并归一化到 `[-1, 1]`。随机裁剪先采样整数 LR 坐标，再按 `data.scale` 映射到 HR 与 Ref 坐标，以保持空间对齐。`RefPNGDataset` 读取的是上述三目录配对格式，不能直接读取 UC Merced/AID 的原始分类目录。

### UC Merced 与 AID

这两个公开数据集原本用于遥感场景分类，并不提供 SR 的 LR/HR 配对。仓库中的准备脚本先按类别做固定种子的 `70%/15%/15%` 分层切分，再从每张原图中心裁剪 HR，使用 PIL bicubic 生成 LR，并把 LR 再 bicubic 上采样到 HR 尺寸写入 Ref：

```text
Ref = bicubic(LR, HR.size)  # SISR reference
```

已下载的原始压缩包（均位于 `data/remote_sensing/raw/`，该目录已被 git 忽略）：

| 数据集 | 文件 | 原始规模 | 准备后 split（train/val/test）与尺寸（scale=4） |
| --- | --- | ---: | --- |
| UC Merced | `uc_merced_land_use.zip` | 21 类 / 2100 张 | `1470/315/315`；HR/Ref `256x256`，LR `64x64` |
| AID | `aid_scene_classification.zip` | 30 类 / 10000 张 | `7000/1507/1493`；HR/Ref `512x512`，LR `128x128` |

来源：UC Merced 原始数据集 [UCMerced Land Use](http://weegee.vision.ucmerced.edu/datasets/landuse.html)（本地下载使用 [Kaggle 镜像](https://www.kaggle.com/datasets/abdulhasibuddin/uc-merced-land-use-dataset)）；AID 原始数据集 [AID](http://captain.whu.edu.cn/AID/)（本地下载使用 [Kaggle 镜像](https://www.kaggle.com/datasets/jiayuanchengala/aid-scene-classification-datasets)）。准备数据：

```bash
mkdir -p data/remote_sensing/raw
curl -L 'https://www.kaggle.com/api/v1/datasets/download/abdulhasibuddin/uc-merced-land-use-dataset' \
  -o data/remote_sensing/raw/uc_merced_land_use.zip
curl -L 'https://www.kaggle.com/api/v1/datasets/download/jiayuanchengala/aid-scene-classification-datasets' \
  -o data/remote_sensing/raw/aid_scene_classification.zip

mkdir -p data/remote_sensing/raw/ucmerced_extracted data/remote_sensing/raw/aid_extracted
unzip -q data/remote_sensing/raw/uc_merced_land_use.zip -d data/remote_sensing/raw/ucmerced_extracted
unzip -q data/remote_sensing/raw/aid_scene_classification.zip -d data/remote_sensing/raw/aid_extracted

conda run -n rwkv7 python scripts/prepare_remote_sensing.py \
  --dataset ucmerced \
  --source-dir data/remote_sensing/raw/ucmerced_extracted/UCMerced_LandUse/Images \
  --output-dir data/remote_sensing/prepared/UC_Merced \
  --source-archive data/remote_sensing/raw/uc_merced_land_use.zip \
  --workers 8

conda run -n rwkv7 python scripts/prepare_remote_sensing.py \
  --dataset aid \
  --source-dir data/remote_sensing/raw/aid_extracted/AID \
  --output-dir data/remote_sensing/prepared/AID \
  --source-archive data/remote_sensing/raw/aid_scene_classification.zip \
  --workers 8
```

脚本会写出 `metadata.json` 和 `manifest.jsonl`，并校验每个 split 的 RGB、尺寸和三目录同名配对。由于两个数据集的 HR 尺寸不同，训练时要使用对应配置，不能把 UC Merced 的 `hr_size=256` 与 AID 的 `hr_size=512` 混在同一个固定网格实验中。

真实参考数据集 `RefSR_data/ALL_2`（Real-RefRSSRD）与上述合成 SISR 数据不同：其 `HR` 和 `Ref` 是真实 NAIP 影像，`LR` 是真实 Sentinel-2 影像，倍率为 10（480×480 / 48×48）。数据说明见 [`RefSR_data/ALL_2/Real-RefRSSRD.md`](RefSR_data/ALL_2/Real-RefRSSRD.md)。HRMS-SCD 的跨时相 RefSR 转换说明见 [`RefSR_data/HRMS_SCD/RefSR-HRMS.md`](RefSR_data/HRMS_SCD/RefSR-HRMS.md)。

## 训练

在仓库根目录使用 `rwkv7` 环境启动：

```bash
conda run -n rwkv7 python scripts/train_sr_prior.py --config configs/sr_prior_hrms_scd_x4.yaml
conda run -n rwkv7 python scripts/train_sr_prior.py --config configs/sr_prior_real_refrssrd_x10.yaml
conda run -n rwkv7 python scripts/train_sr_prior.py --config configs/sr_prior_ucmerced.yaml
conda run -n rwkv7 python scripts/train_sr_prior.py --config configs/sr_prior_aid.yaml
```

| 配置 | HR patch | LR patch | 倍率 | 内部网格 | batch size |
| --- | ---: | ---: | ---: | ---: | ---: |
| `configs/sr_prior_hrms_scd_x4.yaml` | 512 | 128 | 4 | 128 | 4 |
| `configs/sr_prior_real_refrssrd_x10.yaml` | 480 | 48 | 10 | 120 | 4 |
| `configs/sr_prior_ucmerced.yaml` | 256 | 64 | 4 | 64 | 4 |
| `configs/sr_prior_aid.yaml` | 512 | 128 | 4 | 128 | 4 |

表中配置均使用 BF16 混合精度与梯度累积；两个新增配置将验证采样上限设为 `300`，训练集和测试集仍保留全部样本。UC Merced 配置的 HR 网格较小，适合先做快速验证。RTX 5060 Ti（16 GiB）已完成上述配置的单次 batch=4 前向、反向与优化器更新。

训练模块由 L1 损失与可选 SSIM、FFT 损失组成；EMA 仅在真实 optimizer step 后更新，并在验证和测试期间临时应用。学习率默认使用 `ReduceLROnPlateau`，调度器在每次验证运行结束时读取本次聚合后的 `val_loss`，而不是只在 epoch 末读取一次。

### 学习率调度

上述训练配置都使用以下 plateau 规则（具体初始学习率、衰减因子和下限以对应 YAML 为准）：

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
  --config configs/sr_prior_hrms_scd_x4.yaml \
  --load_weights /path/to/weights.ckpt
```

断点续训示例：

```bash
conda run -n rwkv7 python scripts/train_sr_prior.py \
  --config configs/sr_prior_hrms_scd_x4.yaml \
  --resume checkpoints/refrwkv_sr_hrms_scd_x4/last.ckpt
```

## 评测

`scripts/eval_four_settings.py` 可在同一测试集上对比四种输入：`bicubic`（插值基线）、`sisr_ref`（LR 的 bicubic 上采样，SISR 模式）、`dataset_ref`（当前样本的配对真实 Ref）和 `perfect_ref`（HR 作为诊断上限）。HRMS-SCD 应使用 `dataset_ref`；UC Merced 与 AID 是合成 SISR 数据，应使用 `sisr_ref`。`--hr_size` 必须与 checkpoint 训练时的 HR patch 一致，而不是测试图像的边长。

HRMS-SCD 的两个正式测试 split 可按以下命令复现：

```bash
conda run -n rwkv7 python scripts/eval_four_settings.py \
  --ckpt checkpoints/refrwkv_sr_4/last.ckpt \
  --data RefSR_data/HRMS_SCD \
  --splits test_easy test_hard \
  --settings bicubic dataset_ref sisr_ref \
  --scale 4 \
  --hr_size 512 \
  --batch-size 4
```

准备好的遥感数据只有 `train/val/test`，评测时显式指定 `--splits test`：

```bash
conda run -n rwkv7 python scripts/eval_four_settings.py \
  --ckpt checkpoints/refrwkv_sr_hrms_scd_x4/last.ckpt \
  --data data/remote_sensing/prepared/AID \
  --splits test \
  --scale 4 \
  --hr_size 512
```

当前 `checkpoints/refrwkv_sr_4/last.ckpt` 使用 512 网格，因此它交叉评测 UC Merced 时仍应传 `--hr_size 512`；只有使用 UC Merced 专用 checkpoint 时才传 `--hr_size 256`。评测默认加载 EMA 权重，并在推理前调用 `prepare_for_inference()`；`perfect_ref` 仅用于诊断参考信息的理论上限。

## 扩散阶段集成

`scripts/train_sd2_gan.py` 构建 SR prior 时优先读取 `model.sr.hr_size`；未设置时使用 `data.patch_size`。这使 SR 内部网格与扩散训练 patch 保持一致。需要有意使用不同网格时，可在 `model.sr.hr_size` 中显式指定。
