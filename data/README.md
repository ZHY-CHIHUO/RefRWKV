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

`reference_mode: lr_up` 只用 LR 生成自参考，`reference_mode: paired` 才读取磁盘 `Ref/`。测试和对比必须保持同一 split 的 `sample_id` 配对。

常规 Dataset 会保留文件本身的通道数，不再统一转换为 RGB。`PNG/JPEG/BMP/TIFF` 和 NumPy `.npy` 均可作为 `HR`、`LR` 或 `Ref` 文件；灰度 PAN 返回 1 通道，多波段 TIFF/NPY 返回对应通道数。整数栅格默认按 dtype 满量程归一化到 `[-1, 1]`，传感器或浮点数据可在配置中设置 `data.value_scale`，必要时用 `data.clip_range: false` 保留范围外值。NPY 约定为 `HWC`，栅格读取器对常见 `C,H,W` TIFF 会自动转置。

## Wuhan

`data/refsr/Wuhan-dataset/` 是四通道、已配准到同一像素网格的 temporal-pair TIFF 数据。网络倍率为 `1`，`3.75` 只用于 ERGAS 的物理分辨率项。完整网格测试使用重叠 tile，具体配置在 `configs/runs/refsrwkv/wuhan.yaml`。

数据准备脚本位于 `scripts/prepare/`，不要把下载数据或生成缓存提交到仓库。
