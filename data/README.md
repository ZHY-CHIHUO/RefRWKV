# `data/`

这里保存数据加载代码、数据集元信息和本地数据目录。图像、TIFF、压缩包等本地数据被 `.gitignore` 忽略，代码和说明文件保留在仓库中。

## 目录和文件

| 路径 | 内容 |
|---|---|
| `dataset.py` | SR/RefSR 共用的 HR、LR、Ref 图像 Dataset。 |
| `loaders.py` | 按任务构造 train/val/test DataLoader。 |
| `sr/` | 单图 SR 数据集入口和 `data/sr/<dataset>/<split>/` 数据。 |
| `refsr/` | 真实参考图 RefSR、Wuhan temporal-pair 数据集入口。 |
| `raw/` | 原始 SR 压缩包和解压缓存。 |
| `archives/` | RefSR 原始归档。 |

## 常规图像契约

单图 SR 使用：

```text
data/sr/<dataset>/<split>/{HR[,LR]}/<same-name>.*
```

真实参考图 RefSR 使用：

```text
data/refsr/<dataset>/<split>/{HR,LR,Ref}/<same-name>.*
```

`train`、`val`、`test`、`test_easy`、`test_hard` 是数据 split。Bicubic 数据可以由 HR 在内存中生成 LR；传感器数据必须读取原生 LR，不能为了匹配整数倍率随意重采样。

`reference_mode: none` 用于 RefSRWKV 的纯 SR 路径：数据集只返回 `LR/HR`，模型不构造或注入 Ref；`reference_mode: lr_up` 是旧的 LR bicubic 自参考方式，`reference_mode: paired` 才读取磁盘 `Ref/`。测试和对比必须保持同一 split 的 `sample_id` 配对。

常规 Dataset 会保留文件本身的通道数，不再统一转换为 RGB。`PNG/JPEG/BMP/TIFF` 和 NumPy `.npy` 均可作为 `HR`、`LR` 或 `Ref` 文件；灰度 PAN 返回 1 通道，多波段 TIFF/NPY 返回对应通道数。整数栅格默认按 dtype 满量程归一化到 `[-1, 1]`，传感器或浮点数据可在配置中设置 `data.value_scale`，必要时用 `data.clip_range: false` 保留范围外值。NPY 约定为 `HWC`，栅格读取器对常见 `C,H,W` TIFF 会自动转置。

## PanCollection H5

PanCollection 的 `train_*.h5`、`valid_*.h5` 和 reduced-resolution 测试 H5 使用统一的
NCHW 键：`gt` 是 HR、`ms` 是原生低分辨率多光谱、`pan` 是高分辨率单通道参考，
`lms` 是仅供上游工具箱使用的 bicubic 上采样基线。本项目统一加载为：

```text
lr  <- ms       # 例如 WV3: 8 x 16 x 16
ref <- pan      # 例如 WV3: 1 x 64 x 64
hr  <- gt       # 例如 WV3: 8 x 64 x 64
```

`data/refsr/pancollection.py` 是通用 H5 Dataset；通道数、DN 归一化、文件名和 H5 键
由数据集 YAML 提供，不需要为 GF2/QB/WV2 复制加载代码。WV3 的配置和训练入口分别是
`configs/datasets/refsr/pancollection_wv3.yaml`、
`configs/runs/refsrwkv/pancollection_wv3.yaml` 和
`scripts/shortcuts/train/refsrwkv_pancollection_wv3.sh`。远程 H5 文件不提交到仓库，
运行前确认 YAML 中的 `root` 在当前服务器可访问。

## Wuhan

`data/refsr/Wuhan-dataset/` 是四通道、已配准到同一像素网格的 temporal-pair TIFF 数据。网络倍率为 `1`，`3.75` 只用于 ERGAS 的物理分辨率项。完整网格测试使用重叠 tile，具体配置在 `configs/runs/refsrwkv/wuhan.yaml`。

数据准备脚本位于 `scripts/prepare/`，不要把下载数据或生成缓存提交到仓库。
