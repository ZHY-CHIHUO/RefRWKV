# HRMS-SCD x4 对比基线

本页定义 RefRWKV 的固定对比协议及十个可运行基线。范围是：六个单图
SR（SISR）模型和四个真实参考图 RefSR 模型；**不包含 C2-Matching**。
所有 native run 都采用同一份
[`configs/common/benchmark.yaml`](../../configs/common/benchmark.yaml)：

| 项 | 固定值 |
| --- | --- |
| 学习率 / 优化器 | `1e-4` / AdamW，`weight_decay=0`，`betas=[0.9, 0.999]` |
| 调度 | `ReduceLROnPlateau`，`factor=0.5`，`patience=3`，`min_lr=1e-7` |
| 预算 | `max_epochs: -1`，`max_steps: 50000` |
| 验证 | 每 1 epoch（`val_check_interval: 1.0`，`check_val_every_n_epoch: 1`） |
| 提前停止 | `early_stopping_patience: null` |
| 重建损失 | L1；SSIM / FFT 辅助项均显式为 0 |
| 其他共同项 | `accumulate_grad_batches=2`、EMA `0.999`、LR patch `48`、验证最多 300 个样本 |

`max_epochs: -1` 必须和 `max_steps: 50000` 一起保留：前者取消 epoch
上限，后者才是可比较的实际停止条件。训练数据有多少、batch 如何设置都不会
将某一模型提前截断在不同 epoch 数。

## 方法、实现边界和入口

“native compatibility”表示本仓库可在 `rwkv7` 主环境直接训练的实现：它保留
论文方法族的关键机制与统一张量接口，但并不是上游 checkpoint 的逐层兼容副本。
若论文数字或官方 checkpoint 对齐是目标，应使用“官方桥接环境”和原作者代码；
两类结果不能在同一列里混报。

| 任务 | 方法 | 核心思路 | 本仓库实现 | 参数量 / FP32 权重 | 配置 | 训练或评估入口 | 官方仓库 |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| SR | Bicubic | 三次插值 | `reference_only`，无参数、仅评估 | 0 / 0 MB | `configs/runs/sr/bicubic/hrms_scd_x4.yaml` | `scripts/test/sr.py`（不传 checkpoint） | — |
| SR | EDSR | 残差块与 PixelShuffle | native compatibility | 1,517,571 / 6.07 MB | `configs/runs/sr/edsr/hrms_scd_x4.yaml` | `scripts/train/sr.py` | [EDSR](https://github.com/sanghyun-son/EDSR-PyTorch) |
| SR | RCAN | residual-in-residual channel attention | native compatibility | 15,592,355 / 62.37 MB | `configs/runs/sr/rcan/hrms_scd_x4.yaml` | `scripts/train/sr.py` | [RCAN](https://github.com/yulunzhang/RCAN) |
| SR | SwinIR-M | 窗口 Transformer | 本项目现有 SwinIR | 11,900,199 / 47.60 MB | `configs/runs/sr/swinir/hrms_scd_x4.yaml` | `scripts/train/sr.py` | [SwinIR](https://github.com/JingyunLiang/SwinIR) |
| SR | HAT | hybrid / channel attention | native compatibility | 1,339,107 / 5.36 MB | `configs/runs/sr/hat/hrms_scd_x4.yaml` | `scripts/train/sr.py` | [HAT](https://github.com/XPixelGroup/HAT) |
| SR | MambaIRv2 | state-space image restoration | native compatibility；官方 MambaSSM 单独环境 | 701,443 / 2.81 MB | `configs/runs/sr/mambairv2/hrms_scd_x4.yaml` | `scripts/train/sr.py` | [MambaIR](https://github.com/csguoh/MambaIR) |
| RefSR | TTSR | 匹配后纹理传输 | native compatibility；低分辨率局部匹配 + 高分辨率纹理门控 | 1,328,628 / 5.31 MB | `configs/runs/refsr/ttsr/hrms_scd_x4.yaml` | `scripts/train/refsr.py` | [TTSR](https://github.com/researchmm/TTSR) |
| RefSR | MASA-SR | matching acceleration + spatial adaptation | native compatibility；粗匹配 + 可学习位移适配 | 1,399,414 / 5.60 MB | `configs/runs/refsr/masa_sr/hrms_scd_x4.yaml` | `scripts/train/refsr.py` | [MASA-SR](https://github.com/JIA-Lab-research/MASA-SR) |
| RefSR | DATSR | deformable attention reference alignment | native compatibility；`grid_sample` 替代官方 DCNv2 | 1,330,744 / 5.32 MB | `configs/runs/refsr/datsr/hrms_scd_x4.yaml` | `scripts/train/refsr.py` | [DATSR](https://github.com/caojiezhang/DATSR) |

参数量由当前默认 x4 config 实测（`sum(p.numel())`）；FP32 权重仅为参数本身
的 `params × 4`，不包含 optimizer state、EMA 或 activation 显存。不同论文的
官方“大/中/小”配置不等参数，不能把这里的参数量当作原论文模型大小。

## 数据和公平比较

| 组别 | 允许输入 | HRMS-SCD 配置 | 说明 |
| --- | --- | --- | --- |
| SR | `LR → SR` | `swinir`、`edsr`、`rcan`、`hat`、`mambairv2`、`bicubic` | 不读取 `Ref/`；用于衡量纯 SISR 上限。 |
| RefSR | `LR + T1 Ref → SR(T2)` | `refsrwkv/hrms_scd_trefsr_x4`、`ttsr`、`masa_sr`、`datsr` | 强制 `data.reference_mode: paired`，读取跨时相真实参考图。 |
| RefSRWKV-SR 辅助对照 | `LR + bicubic(LR) → SR` | `refsrwkv/hrms_scd_sr_x4` | 不是物理参考图 RefSR；用于分离 RefSRWKV 结构本身与真实参考信息的增益。 |

训练时只用 train；按相同的 val cadence 选择最佳 `val/loss` checkpoint；报告时对
`test_easy` 和 `test_hard` 分开运行、分别报告 PSNR / SSIM。不得在 test 上挑
checkpoint、调学习率或决定停止步数。Bicubic 没有训练和 checkpoint：它只需在两
个 test split 各运行一次。

示例：

```bash
# 可训练 SR
bash scripts/submit_train.sh rcan_hrms_scd_sr_x4.sh

# 可训练真实参考图 RefSR
bash scripts/submit_train.sh datsr_hrms_scd_ref_x4.sh

# Bicubic：显式选择 HRMS-SCD 的测试 split
python scripts/test/sr.py \
  --config configs/runs/sr/bicubic/hrms_scd_x4.yaml \
  --split test_easy
```

快捷脚本可由 `bash scripts/submit_train.sh --list` 查看。所有 trainable run 的
checkpoint 文件名使用 `epoch=0000-step=000596.ckpt` 形式，不会重复
`epoch=` / `step=` 前缀；日志目录按 run 隔离。

每个评估目录的 `metrics.json` 会写入 `implementation`、`variant` 和
`reference_mode`；汇总结果时必须保留这些字段，避免把 `native_compatibility`
和 `official bridge` 的数值混在同一行。

## 复现难度与环境

| 方法 | 主环境可跑 | 官方复现难度 | 原因 / 建议 |
| --- | --- | --- | --- |
| Bicubic | 是 | 很低 | 无训练、无依赖。 |
| EDSR / RCAN | 是 | 低 | 经典 PyTorch；官方代码较旧，原论文配置和本项目固定 50k-step 协议应分开记录。 |
| SwinIR | 是 | 低 | 当前项目已有实现；窗口大小与输入 patch 要保持可整除或使用其 padding 逻辑。 |
| HAT | 是 | 中 | 官方依赖 BasicSR / timm；可独立安装，训练显存通常高于这里的 compact compatibility profile。 |
| MambaIRv2 | 是 | 中到高 | `mamba-ssm`、`causal-conv1d` 编译 CUDA 扩展；请使用独立环境。 |
| TTSR | 是 | 中 | 官方训练依赖 VGG19 特征权重和旧 torchvision 习惯；本项目版本刻意不下载 VGG。 |
| MASA-SR | 是 | 中 | 官方常见流程为先 reconstruction 再 GAN；公平的 L1 对比应只跑 reconstruction 阶段。 |
| DATSR | 是 | 高 | 官方要求 PyTorch 1.7.1 / CUDA 10.0–10.1 / MMCV 0.4.4 / DCNv2；当前 RTX 5060 Ti + CUDA 12.8 上不承诺可编译。 |

独立环境定义在 [`environments/`](../../environments/)，详细安装边界见
[`environments/README.md`](../../environments/README.md)。本次没有在机器上
直接创建 legacy 环境：DATSR 的旧 CUDA 栈和当前驱动/显卡不兼容风险很高，盲目
创建会占用大量空间且通常无法完成 DCNv2 编译。YAML 环境文件可在匹配的容器或
旧 GPU 节点上明确创建。

## 结果表模板

| 类别 | 方法 | 实现标签 | checkpoint / step | test_easy PSNR / SSIM | test_hard PSNR / SSIM | 备注 |
| --- | --- | --- | --- | --- | --- | --- |
| SR | Bicubic | reference_only | — | — | — | 固定插值下限。 |
| SR | EDSR | native_compatibility | 50,000 | — | — |  |
| SR | RCAN | native_compatibility | 50,000 | — | — |  |
| SR | SwinIR-M | native | 50,000 | — | — |  |
| SR | HAT | native_compatibility | 50,000 | — | — |  |
| SR | MambaIRv2 | native_compatibility | 50,000 | — | — |  |
| RefSR | RefSRWKV | native | 50,000 | — | — | paired T1 reference。 |
| RefSR | TTSR | native_compatibility | 50,000 | — | — |  |
| RefSR | MASA-SR | native_compatibility | 50,000 | — | — |  |
| RefSR | DATSR | native_compatibility | 50,000 | — | — |  |

若改用作者官方代码，实施标签改为 `official bridge`，并单独填写实际环境、commit、
训练阶段和数据预处理；不要覆盖 native compatibility 的结果行。
