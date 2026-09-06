# RefRWKV 项目文件导览

RefRWKV 是参考图超分辨率（RefSR）和单图超分辨率（SISR）实验项目。一次训练通常按下面的路径流动：

```text
YAML 配置 → data/loaders.py → 模型 registry → engine → experiments/
                                      ├─ evaluation/
                                      └─ weights/（需要导出时）
```

## 根目录文件

| 路径 | 功能 |
|---|---|
| `README.md` | 安装、配置、训练、测试和数据准备的快速入口。 |
| `requirements.txt` | SR、RefSRWKV 训练和评测所需的基础 Python 依赖。 |
| `requirements-refdiff.txt` | RefDiffRWKV、离线感知指标和判别器所需的额外依赖。 |
| `.gitignore` | 忽略本地数据、缓存、checkpoint、日志和编译产物。 |
| `.vscode/settings.json` | VS Code 的项目编辑器设置。 |
| `scripts/submit.sh` | 远程作业提交的通用 shell 入口。 |
| `scripts/submit_train.sh` | 远程训练兼容入口，可选择 `scripts/shortcuts/` 中的快捷脚本。 |

## `configs/`

YAML 配置按职责拆分。`common` 提供默认值，`datasets` 描述数据，`models` 描述网络，`runs` 描述一次可执行实验。

### `configs/common/`

| 文件 | 功能 |
|---|---|
| `base.yaml` | 设备、batch、优化器、scheduler、EMA 和输出目录的公共默认值。 |
| `sr.yaml` | 单图 SR 的任务默认值，指定 SwinIR 和 `lr/hr` 数据契约。 |
| `refsr.yaml` | RefSR 的通用训练参数。 |
| `refsr_paired.yaml` | 使用磁盘 `Ref/` 的参考图数据契约和参考图增强参数。 |
| `refsrwkv.yaml` | RefSRWKV 的优化器、训练周期和损失默认值。 |
| `refsrwkv_paired.yaml` | RefSRWKV 与真实参考图配对训练时的组合配置。 |
| `refdiffrwkv.yaml` | RefDiffRWKV 的扩散、先验、判别器和采样默认值。 |

### `configs/datasets/`

每个文件记录数据根目录、split 数量、图像尺寸、LR 的物理来源和默认倍率。

| 文件 | 功能 |
|---|---|
| `datasets/sr/aid.yaml` | AID 的 bicubic x4 SISR 数据元信息。 |
| `datasets/sr/ucmerced.yaml` | UC Merced 的 bicubic x4 SISR 数据元信息。 |
| `datasets/refsr/hrms_scd.yaml` | HRMS-SCD 跨时相参考数据元信息，LR 为 HR bicubic x4。 |
| `datasets/refsr/real_refrssrd.yaml` | Real-RefRSSRD 真实传感器数据元信息，LR 为原生 x10 Sentinel-2 观测。 |

### `configs/models/`

| 文件 | 功能 |
|---|---|
| `models/sr/swinir_m.yaml` | SwinIR-M 的窗口、深度、通道和上采样参数。 |
| `models/refsr/refsrwkv.yaml` | RefSRWKV 的编码器、匹配窗口、全局 latent、decoder refusion 和置信度门控开关。 |
| `models/refsr/refdiffrwkv.yaml` | RefDiffRWKV 的模型族标识，并引入扩散公共配置。 |

### `configs/runs/`

run 文件把任务、数据和模型组合成可直接启动的实验。

| 路径 | 功能 |
|---|---|
| `runs/sr/swinir/aid_x4.yaml` | SwinIR 在 AID x4 上的训练配置。 |
| `runs/sr/swinir/ucmerced_x4.yaml` | SwinIR 在 UC Merced x4 上的训练配置。 |
| `runs/sr/swinir/hrms_scd_x4.yaml` | SwinIR 在 HRMS-SCD x4 上的 SISR 配置。 |
| `runs/refsrwkv/aid_x4.yaml` | RefSRWKV 在 AID 上使用 bicubic LR 自参考的配置。 |
| `runs/refsrwkv/ucmerced_x4.yaml` | RefSRWKV 在 UC Merced 上使用 bicubic LR 自参考的配置。 |
| `runs/refsrwkv/hrms_scd_sr_x4.yaml` | RefSRWKV 在 HRMS-SCD 上使用 LR 上采样自参考的 SISR 配置。 |
| `runs/refsrwkv/hrms_scd_x4.yaml` | RefSRWKV 在 HRMS-SCD 三元组上的训练配置。 |
| `runs/refsrwkv/hrms_scd_trefsr_x4.yaml` | HRMS-SCD 跨时相 TRefSR 的清晰命名入口。 |
| `runs/refsrwkv/real_refrssrd_x10.yaml` | RefSRWKV 在 Real-RefRSSRD x10 上的训练配置。 |
| `runs/refdiffrwkv/stage1.yaml` | 扩散系统基础训练阶段。 |
| `runs/refdiffrwkv/stage2.yaml` | 加入语义和 SR 条件的训练阶段。 |
| `runs/refdiffrwkv/stage3.yaml` | 加入置信度、时序门控和自相似传播的训练阶段。 |
| `runs/refdiffrwkv/stage4.yaml` | 启用判别器和 GAN 损失的完整训练阶段。 |

## `data/`

### Python 数据代码

| 文件 | 功能 |
|---|---|
| `data/dataset.py` | `SuperResolutionDataset`：读取对齐的 HR/LR/Ref PNG，按 `return_items` 返回 `lr/hr` 或 `lr/hr/ref`，并按配置生成 bicubic LR 或自参考。 |
| `data/loaders.py` | 根据任务配置构造 train/val/test DataLoader，处理目录聚合、split 和采样上限。 |
| `data/sr/dataset.py` | SR 任务命名入口，调用统一 Dataset。 |
| `data/refsr/dataset.py` | RefSR 任务命名入口，调用统一 Dataset。 |
| `data/sr/__init__.py`、`data/refsr/__init__.py` | 导出对应任务的数据集接口。 |

### 数据目录和说明

| 路径 | 功能 |
|---|---|
| `data/sr/AID/` | AID 的 HR/LR split 数据（LR 可由 HR bicubic x4 生成）。 |
| `data/sr/UC_Merced/` | UC Merced 的 HR/LR split 数据（LR 可由 HR bicubic x4 生成）。 |
| `data/refsr/HRMS_SCD/` | HRMS-SCD 的 HR/LR/Ref split 数据和元信息。 |
| `data/refsr/Real-RefRSSRD/` | Real-RefRSSRD 的 HR/LR/Ref split 数据和数据说明。 |
| `data/refsr/CUFED/`、`data/refsr/CUFED5/` | 参考数据集的本地数据目录标记。 |
| `data/raw/` | 原始数据压缩包或解压内容。 |
| `data/archives/` | 数据归档文件。 |
| `data/refsr/HRMS_SCD/RefSR-HRMS.md` | HRMS-SCD 的来源、构建规则、split 和使用示例。 |
| `data/refsr/HRMS_SCD/meta.json` | HRMS-SCD 的尺寸、倍率、双向样本和 split 统计。 |
| `data/refsr/Real-RefRSSRD/Real-RefRSSRD.md` | Real-RefRSSRD 的传感器、尺寸、split 和评测说明。 |

每个 split 的最小结构是 `HR/`；bicubic 数据的 `LR/` 可以省略，真实传感器数据必须提供 `LR/`；只有 `paired` RefSR 需要 `Ref/`。

## `models/`

### 公共入口

| 文件 | 功能 |
|---|---|
| `models/sr/registry.py` | 注册和构造 SR 模型 adapter。 |
| `models/refsr/registry.py` | 注册和构造 RefSR 模型。 |
| `models/sr/__init__.py`、`models/refsr/__init__.py` | 暴露模型构造接口。 |

### `models/sr/swinir/`

| 文件 | 功能 |
|---|---|
| `network.py` | Swin Transformer、RSTB、窗口注意力和上采样网络。 |
| `adapter.py` | 将配置、输入值域和输出张量接到 SR registry。 |
| `__init__.py` | 导出 SwinIR 组件。 |

### `models/refsr/refsrwkv/`

| 文件 | 功能 |
|---|---|
| `model.py` | RefSRWKV 主网络，包括 LR/Ref 编码、局部匹配、全局 latent 和解码器。 |
| `adapter.py` | 将 RefSRWKV 接到 RefSR registry，并处理 scale 和配置。 |
| `__init__.py` | 导出 RefSRWKV 组件。 |

### `models/refsr/RefDiffRWKV/`

| 文件 | 功能 |
|---|---|
| `RefDiffRWKV.py` | RWKV 参考特征编码器和自相似纹理传播模块。 |
| `modules.py` | 卷积块、注意力、下采样、参考编码器等基础模块。 |
| `sd2_ref_adapter.py` | 将 LR/Ref 特征注入 Stable Diffusion 2 UNet。 |
| `sd2_ref_generator.py` | VAE、扩散 scheduler、UNet 前向、采样和像素解码。 |
| `sd2_ref_gan_system.py` | 扩散生成器、SR prior 和优化器的训练系统。 |
| `sd2_ref_discriminator.py` | 语义判别器和纹理一致性判别器。 |
| `globalsemanticmodule.py` | DINOv2 特征、SR 条件和 RWKV 语义金字塔。 |
| `cond_fn.py` | 采样阶段的条件引导和梯度计算。 |
| `spaced_sampler.py` | IDDPM 间隔采样、噪声调度和采样步管理。 |
| `__init__.py` | 导出扩散模型组件。 |

## `engines/`

| 文件或目录 | 功能 |
|---|---|
| `engines/base_trainer.py` | 公共训练生命周期、优化器、EMA、验证、checkpoint 和 scheduler。 |
| `engines/sr/trainer.py` | SR 的 batch 解包、前向、损失和验证步骤。 |
| `engines/refsr/refsrwkv_trainer.py` | RefSRWKV 的 batch、参考图和损失步骤。 |
| `engines/refsr/refdiff_trainer.py` | RefDiffRWKV 的生成器/判别器交替训练和扩散损失。 |
| `*/__init__.py` | 导出各任务的 engine。 |

## `runtime/`

| 文件 | 功能 |
|---|---|
| `runtime/config.py` | 读取、合并、展开和校验 YAML 配置。 |
| `runtime/checkpoint.py` | 保存、读取和匹配模型权重、EMA、配置与实验签名。 |
| `runtime/experiments.py` | 生成训练/测试目录并保存配置快照。 |
| `runtime/common.py` | 路径解析、随机种子和通用 EMA 等运行时工具。 |
| `runtime/__init__.py` | 导出运行时接口。 |

## `kernels/`

| 文件 | 功能 |
|---|---|
| `kernels/wkv/bi_wkv.cpp` | WKV CUDA 扩展的 C++ 绑定。 |
| `kernels/wkv/bi_wkv_kernel.cu` | WKV 前向和反向 CUDA kernel。 |
| `kernels/wkv/layers.py` | WKV 的 PyTorch 层封装。 |
| `kernels/wkv/runtime.py` | 编译、加载和调用 WKV 后端。 |
| `kernels/wkv/__init__.py` | 导出 WKV 接口。 |

## `losses/`、`metrics/`、`evaluation/`

| 路径 | 功能 |
|---|---|
| `losses/common.py` | L1、SSIM、FFT 等可组合损失。 |
| `metrics/__init__.py` | 指标包入口；基础 PSNR/SSIM 由评测流程调用。 |
| `evaluation/runner.py` | 构造 loader 和模型，运行推理，保存图片和指标。 |
| `evaluation/eval_pyiqa.py` | 使用 PyIQA 计算 LPIPS、DISTS 等感知指标。 |
| `evaluation/eval_sewar.py` | 使用 sewar 计算传统图像质量指标。 |

## `scripts/`

| 文件或目录 | 功能 |
|---|---|
| `scripts/train/sr.py` | 启动 SwinIR 等 SR 训练。 |
| `scripts/train/refsrwkv.py` | 启动 RefSRWKV 训练。 |
| `scripts/train/refdiffrwkv.py` | 启动 RefDiffRWKV 训练。 |
| `scripts/test/sr.py` | SR 推理和测试。 |
| `scripts/test/refsr.py` | RefSRWKV/RefDiffRWKV 推理和测试。 |
| `scripts/evaluate.py` | 统一评估入口。 |
| `scripts/prepare/remote_sensing.py` | 将遥感原始图像整理为 HR/LR split，并生成 bicubic LR。 |
| `scripts/shortcuts/` | 按模型、数据集、任务和倍率命名的训练快捷入口；`show_commands.sh` 可集中查看命令。 |
| `scripts/submit.sh`、`scripts/submit_train.sh` | 远程作业和训练提交入口。 |

## `tests/`

| 文件 | 功能 |
|---|---|
| `tests/test_unified_dataset.py` | 统一 Dataset 的目录、倍率、返回字段和增强测试。 |
| `tests/test_reference_contract.py` | `lr_up`、`paired` 和 Ref 读取规则测试。 |
| `tests/test_refsrwkv_ablation.py` | RefSRWKV 消融开关和配置校验测试。 |
| `tests/smoke_native_geometry.py` | CUDA WKV 不同倍率下的形状和梯度冒烟测试。 |
| `tests/__init__.py` | 测试包入口。 |

## 运行产出和文档

| 路径 | 功能 |
|---|---|
| `experiments/train/` | 训练 checkpoint、TensorBoard 日志和配置快照。 |
| `experiments/test/` | 测试图片和 `metrics.json`。 |
| `experiments/.gitkeep` | 保持实验输出根目录存在。 |
| `weights/pretrained/` | 外部或共享的预训练权重。 |
| `weights/exports/` | 明确导出的部署权重。 |
| `weights/README.md` | 权重目录和 checkpoint 存放规则。 |
| `docs/architecture.md` | 目录、数据契约、配置分层和扩展约定。 |
| `docs/models/` | 模型结构和模型级实验说明的文档位置。 |
| `docs/training/` | 训练命令、消融方案和运行记录的文档位置。 |
| `docs/project_guide.md` | 项目各目录和关键文件的功能索引。 |

`docs/models/` 和 `docs/training/` 用于按模型和训练任务分类文档；具体说明放入对应目录。
