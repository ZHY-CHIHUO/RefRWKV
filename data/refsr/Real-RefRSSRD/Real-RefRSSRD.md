# Real-RefRSSRD：真实世界参考超分数据集

Real-RefRSSRD（Real-World Reference-based Super-Resolution Dataset）由 CRefDiff 工作提出，面向真实遥感观测中的参考引导超分辨率（RefSR）。它把当前时刻的低分辨率观测、当前时刻的高分辨率目标和历史高分辨率参考放在同一空间位置配对：

```text
当前 Sentinel-2 LR + 历史 NAIP Ref -> 当前 NAIP HR
```

与把 HR 用 bicubic 人工降采样得到 LR 的合成数据不同，Real-RefRSSRD 的 LR 来自真实 Sentinel-2 观测，包含跨传感器分辨率、辐射和噪声差异。

## 1. 原始数据属性

| 角色 | 传感器/来源 | 采集时间 | GSD | 论文中的 patch |
|---|---|---|---:|---:|
| **HR（目标）** | NAIP | 2020–2023 | 1 m | 480 × 480 |
| **Ref（历史参考）** | NAIP | 2009–2015 | 1 m | 480 × 480 |
| **LR（当前观测）** | Sentinel-2 | 2020–2023 | 10 m | 48 × 48 |

配对约束：

- LR 与 HR 在空间上对齐，采集时间差小于 7 天，LR 云覆盖率低于 10%。
- Ref 与 HR 空间对齐，但时间间隔为 6–14 年，且多数超过 10 年，因此会出现建筑、植被和土地利用变化。
- 数据覆盖美国 43 个场景，包含居民区、工业区、农田、森林、水体、操场和道路等地物；采样更偏向城市区域。
- 超分倍率为 **10×**：48 × 48 LR 对应 480 × 480 HR/Ref。

## 2. 数据集划分

| split | 样本数 | 空间位置 |
|---|---:|---|
| `train` | 74,093 | 每个场景的上部区域 |
| `val` | 172 | 每个场景的左下区域 |
| `test` | 1,456 | 每个场景的右下区域 |

划分在每个场景内按空间位置完成，避免相邻 patch 在训练和测试之间泄漏。论文另外按 `Ref` 与 `HR` 的 LPIPS 相似度从测试样本中选取 `L1`、`L2`、`L3`、`L4` 四个分析子集，每个子集 200 对；四个子集是相似度分层评估用的子集，不应替代完整 `test` 的 1,456 对报告。

## 3. 本目录的实际文件

本仓库的 `ALL_2` 是按当前项目 loader 整理好的 PNG 副本，不是原始数据发布页的目录镜像：

```text
data/refsr/Real-RefRSSRD/
├── train/
│   ├── HR/          # 74,093 张，480×480，RGB PNG
│   ├── LR/          # 74,093 张，48×48，RGB PNG（原始低分辨率观测）
│   └── Ref/         # 74,093 张，480×480，RGB PNG（历史 NAIP）
├── val/
│   └── {HR,LR,Ref}/ # 每个目录 172 张
├── test/
│   └── {HR,LR,Ref}/ # 每个目录 1,456 张
└── Real-RefRSSRD.md
```

本地核对结果：三个 split 的 `HR/LR/Ref` 文件名一一对应，文件均为 8-bit RGB PNG，尺寸为 `480/48/480`（HR/LR/Ref）。`Ref` 是真实历史影像，不是 `LR` 的上采样结果。

上游 CRefDiff 代码为了适配其扩散模型，会另行生成 `LR_Ux10`（把 48 × 48 LR 最近邻放大到 480 × 480）。本项目的 `RefPNGDataset` 读取原始 `LR`，在模型内部按 `scale=10` 处理，不需要也不应把 `LR_Ux10` 改名覆盖 `LR`。

## 4. 加载方式

### 4.1 PNG 文件夹模式

```python
from data.refsr.dataset import RefPNGDataset

train_ds = RefPNGDataset(
    data_dir="data/refsr/Real-RefRSSRD",
    mode="train",
    patch_size=480,
    scale=10,
    augment=True,
    augment_ref=True,
    ref_gray_prob=0.2,
)

sample = train_ds[0]
# lr: (3, 48, 48), hr/ref: (3, 480, 480), values in [-1, 1]
```

验证和测试时应关闭随机增强：

```python
val_ds = RefPNGDataset(
    data_dir="data/refsr/Real-RefRSSRD",
    mode="val",
    patch_size=480,
    scale=10,
    augment=False,
    augment_ref=False,
)
```

当前训练和评测统一使用上述 PNG 文件夹模式。PNG、原始压缩包和其他本地数据均由 `.gitignore` 排除，GitHub 只上传本说明和代码。

## 5. 当前 RefSRWKV 训练入口

本项目对应配置为 `configs/runs/refsrwkv/real_refrssrd_x10.yaml`：

```bash
conda run -n rwkv7 python scripts/train/refsrwkv.py \
  --config configs/runs/refsrwkv/real_refrssrd_x10.yaml
```

`configs/runs/refsrwkv/real_refrssrd_x10.yaml` 使用 `run.scale=10`、`run.hr_patch=480`、现成 CUDA Bi-WKV 和 `rwkv7` 环境；训练时启用参考图风格增强（亮度、对比度、饱和度、色相和灰度化）。配置保持 `data.reference_mode: paired`，不会将真实历史 `Ref` 替换为 bicubic LR；若做 SISR 对照，应显式设置 `data.reference_mode=lr_up`，并单独记录该实验。

## 6. 评测注意事项

- 默认测试应使用完整 `test` split；`L1`–`L4` 只用于按参考相似度分层分析。
- 论文中的 PSNR/SSIM 主要在 Y 通道上计算；本项目训练和通用评测脚本使用 RGB 张量时，数值不能直接与论文表格等同比较，必须注明通道和动态范围。
- 评测时保持 `augment=False`、`augment_ref=False`，并确认 checkpoint 的 `scale=10` 与 `hr_size=480`。
- Real-RefRSSRD 是真实跨传感器 RefSR 基准；不要把它与 UC Merced、AID 的 bicubic 合成 SISR 对混合汇报。

## 7. 来源与引用

- 项目代码与数据说明：[CRefDiff](https://github.com/wwangcece/CRefDiff)
- 数据下载页：[Real-RefRSSRD on Hugging Face](https://huggingface.co/datasets/wangcce/Real-RefRSSRD)
- 论文：[arXiv:2506.23801](https://arxiv.org/abs/2506.23801)

建议引用：

```bibtex
@misc{wang2025controllablereferencebasedrealworldremote,
  title         = {Controllable Reference-Based Real-World Remote Sensing Image
                   Super-Resolution with Generative Diffusion Priors},
  author        = {Ce Wang and Wanjie Sun},
  year          = {2025},
  eprint        = {2506.23801},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2506.23801}
}
```

原始 NAIP、Sentinel-2 和 Hugging Face 数据发布条款优先于本项目说明。使用数据时请遵守相应许可，并同时保留数据集论文引用。
