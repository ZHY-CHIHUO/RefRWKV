# RefSR-HRMS：跨时相参考超分数据集

> 基于 [HRMS-SCD](https://github.com/17x-osborn/HRMS-SCD) 构建的 **Reference-based Super-Resolution** 数据集，
> 面向扩散模型驱动的遥感影像超分任务。

---

## 1. 数据集动机

### 1.1 现有 RefSR 数据集的局限

传统 RefSR 数据集（CUFED、Vimeo-90K、COCO-RefSR 等）中，Reference 与 Target 来自**同一张图的不同 crop 或相邻帧**，
两者天然完美匹配（相关性 > 0.95）。这种设定无法反映真实遥感场景中的**跨时相域差异**：

| 差异来源 | 表现 |
|---|---|
| 不同卫星传感器 | 光谱响应函数、MTF、辐射定标参数不同 |
| 不同采集日期 | 大气散射、光照角度、季节物候变化 |
| 真实地物变化 | 建筑新建/拆除、农田翻耕、施工开挖 |

### 1.2 RefSR-HRMS 的定位

RefSR-HRMS 利用 HRMS-SCD 的**双时相配对影像**（T1 = 2017, T2 = 2018），
天然引入上述域差异，为 RefSR 研究提供**更贴近真实遥感应用**的训练与评测环境。

> **核心差异**：Reference（T1）与 Target（T2）来自**不同卫星、不同年份**，
> 通道相关性仅 **~0.48**，远低于传统 RefSR 数据集的 > 0.95。

---

## 2. 数据来源

| 属性 | 值 |
|---|---|
| 原始数据集 | HRMS-SCD（ISPRS Geospatial Week 2025） |
| 影像来源 | 资源一号 (ZY-1)、资源二号 (ZY-2)、北京二号 (BJ-2) |
| 覆盖区域 | 中国北京 |
| 时间范围 | 2017（T1）→ 2018（T2） |
| 空间分辨率 | 1 米 |
| 影像尺寸 | 512 × 512 × 3（uint8, RGB） |
| 总对数 | 11,587 对 |
| 语义类别 | 7 类（种植土地、林草覆盖、建筑物、铁路道路、构筑物、人工挖掘、水体） |

这里的 `RefSR-HRMS` 是在原始 HRMS-SCD 配对影像上整理出的 RefSR 版本；原始 HRMS-SCD 的主要任务仍是语义变化检测。本目录只保留 RefSR 所需的 `HR/LR/Ref` PNG 和 `meta.json`，不包含原始下载包或变化标注。

---

## 3. 角色分配

```
┌────────────────────────────────────────────────────────────────┐
│  T2 (2018, 清晰度 801)                                        │
│  ┌──────────────────┐    bicubic ↓4×    ┌──────────────────┐  │
│  │   HR (GT)        │ ───────────────►  │   LR (Query)     │  │
│  │   512 × 512      │                   │   128 × 128      │  │
│  └──────────────────┘                   └──────────────────┘  │
│                                                                │
│  T1 (2017, 清晰度 516)                                        │
│  ┌──────────────────┐                                         │
│  │   Ref (Reference)│    ← 与 GT 存在天然域差异                │
│  │   512 × 512      │                                         │
│  └──────────────────┘                                         │
└────────────────────────────────────────────────────────────────┘
```

| 角色 | 来源 | 尺寸 | 说明 |
|---|---|---|---|
| **HR (Ground Truth)** | T2 原图 | 512 × 512 × 3 | 目标时相，清晰度更高 |
| **LR (Query)** | T2 bicubic ↓4× | 128 × 128 × 3 | 退化后的目标图像 |
| **Ref (Reference)** | T1 原图 | 512 × 512 × 3 | 参考图，与 GT 存在天然域差异 |

### 退化方式

```
LR = cv2.resize(HR, (128, 128), interpolation=cv2.INTER_CUBIC)
```

仅使用 **bicubic 下采样**，不叠加额外模糊或噪声。
原因：T1-T2 之间已存在巨大的天然域差异（色偏、清晰度差、相关性低），
无需人工退化即可提供充分的任务难度。

---

## 4. 天然域差异统计

| 指标 | 数值 | 含义 |
|---|---|---|
| T1-T2 通道相关性 | **~0.48** | 参考图仅约一半信息与 GT 相关 |
| 系统性色偏 | ΔB = -27, ΔG = -16, ΔR = +22 | T1 偏青绿，T2 偏暖红 |
| 清晰度 (Laplacian Var) | T1 = 516, T2 = 801 | T1 比 T2 模糊 55% |
| 平均变化比例 | 11.28% | 约 1/9 的像素发生了地物变化 |

### 与现有 RefSR 数据集的对比

| 数据集 | Ref 来源 | Ref-GT 相关性 | 主要挑战 |
|---|---|---|---|
| CUFED | 同图不同 crop | > 0.95 | 纹理匹配 |
| Vimeo-90K | 相邻帧 | > 0.90 | 运动估计 |
| **RefSR-HRMS (本数据集)** | **跨时相不同卫星** | **~0.48** | **跨域参考 + 纹理匹配** |

---

## 5. 数据集划分

按 `IMG_Change` 变化掩膜的变化比例划分，以**场景（地理位置）** 为单位，避免数据泄露。下表的场景数、样本数和平均变化比例来自本目录的 `meta.json`；变化范围是构建时的统计摘要。

### 5.1 划分策略

```
变化比例:  0%        2%        5%                          15%       100%
           |─────────|─────────|──────────────────────────|─────────|
           test_easy  val        train                      test_hard
           (500对)   (1058对)  (9529对, 双向=19058样本)    (500对)
           ↓          ↓         ↓                          ↓
           参考图     低变化     全难度梯度                   大面积变化
           高度可靠   监控信号   扩散模型学习鲁棒性           考验鲁棒性
```

### 5.2 各 Split 统计

| Split | 场景数 | 样本数 | 平均变化 | 变化范围 | 用途 |
|---|---|---|---|---|---|
| `train` | 9,529 | **19,058**（双向） | 12.10% | 0.9% – 97.4% | 训练（T1/T2 互做 GT） |
| `val` | 1,058 | 1,058 | 0.40% | 0.0% – 0.9% | 验证（低变化，监控信号稳定） |
| `test_easy` | 500 | 500 | 0.89% | 0.0% – 2.0% | 测试（参考图高度可靠） |
| `test_hard` | 500 | 500 | 29.16% | 15.0% – 98.8% | 测试（参考图大面积不可靠） |
| **总计** | **11,587** | **21,116** | — | — | **利用率 100%** |

### 5.3 双向增强

训练集中每个场景生成**两个样本**：

```
方向 1: GT = T2, Ref = T1  (主方向，GT 清晰度高)
方向 2: GT = T1, Ref = T2  (辅助方向，训练对称性)
```

→ 训练量翻倍（9,529 → 19,058），模型学到参考关系的双向对称性。
文件命名：方向 1 为 `{stem}.png`，方向 2 为 `{stem}_rev.png`。

### 5.4 划分原则

- **训练集覆盖全难度**：变化比例 0.9%–97.4%，让扩散模型学会在参考图不可靠时自适应退化
- **验证集取低变化样本**：保证 val_loss 稳定反映模型真实能力，避免 EarlyStopping 误判
- **测试集分两档**：分别报告 test_easy / test_hard 指标，展示方法在不同参考质量下的表现
- **无场景重叠**：同一地理位置的所有数据只出现在一个 split 中

---

## 6. 目录结构

```
RefSR_data/HRMS_SCD/
├── train/
│   ├── LR/          # 128×128 PNG, RGB (T2 bicubic ↓4×)
│   ├── HR/          # 512×512 PNG, RGB (T2 原图)
│   └── Ref/         # 512×512 PNG, RGB (T1 原图)
├── val/
│   ├── LR/ HR/ Ref/
├── test_easy/
│   ├── LR/ HR/ Ref/
├── test_hard/
│   ├── LR/ HR/ Ref/
└── meta.json        # 构建参数与 split 统计
```

当前目录核对结果：`train/val/test_easy/test_hard` 分别为 `19058/1058/500/500` 对，每个 split 的 `LR`、`HR`、`Ref` 文件名完全一致；所有 PNG 为 8-bit RGB，尺寸分别为 `128×128`、`512×512`、`512×512`。由于训练 split 含双向样本，文件名中的 `_rev` 只在 `train` 中出现。

### 文件命名约定

| 文件名 | 含义 |
|---|---|
| `0.png`, `1.png`, ... | 方向 1（T2 做 GT） |
| `0_rev.png`, `1_rev.png`, ... | 方向 2（T1 做 GT，仅 train） |

---

## 7. 使用方式

### 7.1 通过 RefPNGDataset 加载

```python
from RefSR_data.RefDataset import RefPNGDataset

# 训练（全图 512×512，不裁）
train_ds = RefPNGDataset(
    data_dir="RefSR_data/HRMS_SCD",
    mode="train",
    scale=4,
    patch_size=None,       # 全图，不裁
    augment=True,          # flip + rot90
    augment_ref=False,     # 关闭，天然差异已足够
    ref_gray_prob=0.0,
)

# 验证
val_ds = RefPNGDataset(
    data_dir="RefSR_data/HRMS_SCD",
    mode="val",
    scale=4,
    patch_size=None,
    augment=False,
    augment_ref=False,
)

# 测试（简单 / 困难）
test_easy = RefPNGDataset(
    data_dir="RefSR_data/HRMS_SCD",
    mode="test_easy",
    scale=4, patch_size=None, augment=False, augment_ref=False,
)
test_hard = RefPNGDataset(
    data_dir="RefSR_data/HRMS_SCD",
    mode="test_hard",
    scale=4, patch_size=None, augment=False, augment_ref=False,
)
```

### 7.2 训练 Config 示例

```yaml
# configs/sr_prior_hrms_scd_x4.yaml
data:
  root: "RefSR_data/HRMS_SCD"
  patch_size: 512
  scale: 4
  augment: true
  augment_ref: false
  ref_gray_prob: 0.0
  batch_size: 4
  val_batch_size: 1
  num_workers: 8
  val_num_workers: 2
  prefetch_factor: 4
```

完整训练入口是 `configs/sr_prior_hrms_scd_x4.yaml`；它使用现成 CUDA Bi-WKV、`rwkv7` 环境和每次验证触发的 plateau 学习率调度。训练命令：

```bash
conda run -n rwkv7 python scripts/train_sr_prior.py \
  --config configs/sr_prior_hrms_scd_x4.yaml
```

---

## 8. 对扩散模型 RefSR 的特殊价值

### 8.1 扩散模型天然适配

扩散模型的核心能力是**在不确定性中生成合理内容**。本数据集中：

- **参考图可靠区域**（低变化）→ 模型充分利用纹理迁移
- **参考图不可靠区域**（高变化）→ 模型自适应退化为无条件超分

这种 **"选择性迁移"** 能力是传统回归 / GAN 模型难以学到的，却是扩散模型的天然优势。

### 8.2 论文可报告的多维指标

| 测试集 | 评估重点 | 预期表现 |
|---|---|---|
| `test_easy` | 理想参考下的超分能力 | PSNR / SSIM 最高 |
| `test_hard` | 参考图不可靠时的鲁棒性 | 验证方法不会"强行迁移错误纹理" |

### 8.3 消融实验建议

| 实验 | 说明 |
|---|---|
| 单向 vs 双向 | 对比 `train` 中是否使用 `_rev` 样本 |
| 全难度训练 vs 仅低变化训练 | 对比 train 包含 0-97% vs 仅 <5% 的效果 |
| test_easy vs test_hard | 展示模型在不同参考质量下的泛化能力 |

---

## 9. 构建与复现边界

当前仓库提交的是已经整理好的 RefSR PNG。原始 HRMS-SCD 下载包、变化掩膜和生成这些文件的临时构建脚本不在仓库内；因此不能用本仓库中的命令从零重建原始配对。`meta.json` 保存了当前版本的倍率、尺寸、双向标记和 split 统计，`RefPNGDataset` 可直接读取现有目录。

如果重新整理数据，必须保持以下约定：

- `HR` 与 `Ref` 为同一空间位置的 512×512 RGB PNG，`LR` 为对应 `HR` 的 4× bicubic 下采样（128×128）。
- 训练 split 才生成 `_rev` 双向样本；验证和两个测试 split 只保留主方向。
- 变化场景按地理位置只进入一个 split，不能按 patch 随机打散。

---

## 10. meta.json 字段说明

```json
{
  "source": "HRMS-SCD",
  "scale": 4,
  "hr_size": [512, 512],
  "lr_size": [128, 128],
  "ref_size": [512, 512],
  "degradation": "bicubic",
  "bidirectional": true,
  "splits": {
    "train":     { "scenes": 9529, "samples": 19058, "avg_change_ratio": 0.121 },
    "val":       { "scenes": 1058, "samples": 1058,  "avg_change_ratio": 0.004 },
    "test_easy": { "scenes": 500,  "samples": 500,   "avg_change_ratio": 0.009 },
    "test_hard": { "scenes": 500,  "samples": 500,   "avg_change_ratio": 0.292 }
  }
}
```

---

## 11. 引用

如使用本数据集，请同时引用原始 HRMS-SCD 论文：

```bibtex
@article{guo2025hrms,
  title   = {HRMS-SCD: A High-Resolution Multi-Scene Satellite Imagery
             Dataset for Comprehensive Land-Cover Semantic Change Detection},
  author  = {Guo, Peixin and Yang, Siyu and Zhang, Hanchao and Huang, Xiao
             and Ning, Xiaogang and Han, Yilong and Zhang, Ruiqian and Hao, Minghui},
  journal = {ISPRS Annals of the Photogrammetry, Remote Sensing and
             Spatial Information Sciences},
  volume  = {X-G-2025},
  pages   = {323--331},
  year    = {2025},
  doi     = {10.5194/isprs-annals-X-G-2025-323-2025}
}
```

---

## 12. 许可与致谢

- 原始数据来源于中国测绘科学研究院，用于土地覆盖调查与遥感研究
- 本数据集的 RefSR 转换仅用于学术研究目的
- 感谢 HRMS-SCD 项目团队提供高质量标注数据
