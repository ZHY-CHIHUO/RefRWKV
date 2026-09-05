# RefRWKV 新目录结构

本仓库现在按“任务、数据、模型、运行时产出”拆分。代码目录只放可复用代码，数据、权重、日志和推理结果分别放在自己的生命周期目录中。`SR`（只有 LR/HR）和 `RefSR`（LR/HR/Ref）使用不同的数据契约，加载器不会相互兜底读取。

## 目录总览

```text
RefRWKV/
├── configs/
│   ├── common/                 # 任务公共默认值
│   ├── datasets/{sr,refsr}/    # 数据集元信息与路径
│   ├── models/{sr,refsr}/      # 网络结构默认值
│   └── runs/                   # 可直接启动的实验配置
├── data/
│   ├── sr/                     # 只含 SR 数据集（HR/LR）
│   ├── refsr/                  # 只含 RefSR 数据集（HR/LR/Ref）
│   ├── raw/sr/                 # SR 原始压缩包或解压缓存
│   └── archives/refsr/         # RefSR 原始压缩包
├── models/
│   ├── sr/                     # 单图 SR 模型与 registry
│   │   └── swinir/             # SwinIR 网络和 adapter
│   └── refsr/
│       ├── refsrwkv/           # 参考超分 RWKV 模型
│       └── RefDiffRWKV/        # 参考超分扩散模型及其 G/D 组件
├── engines/
│   ├── base_trainer.py         # 训练公共生命周期
│   ├── sr/trainer.py           # SR 的 train/eval step
│   └── refsr/                  # RefSRWKV 与 RefDiffRWKV engine
├── kernels/wkv/                # WKV CUDA 源码，模型只通过统一接口调用
├── losses/                     # 可组合损失
├── metrics/                    # 指标实现
├── evaluation/                # 推理、指标汇总和结果写盘
├── runtime/                    # 配置、checkpoint、实验路径、EMA 等运行时工具
├── scripts/
│   ├── train/                  # sr.py、refsrwkv.py、refdiffrwkv.py
│   ├── test/                   # SR/RefSR 推理入口
│   ├── prepare/                # 数据准备脚本
│   └── evaluate.py             # 统一评估入口
├── experiments/
│   ├── train/                  # 每次训练的 checkpoint、config、TensorBoard 日志
│   └── test/                   # 每次测试的图片和 metrics.json
├── weights/
│   ├── pretrained/              # 外部或迁移来的预训练权重
│   └── exports/                 # 明确导出的部署权重
├── tests/                      # 轻量 smoke test 与回归测试
└── docs/                       # 架构和数据说明
```

`test` 是测试产出的推荐名称；`test_easy` 和 `test_hard` 是数据集内部的 split，而不是另一个顶层目录。这样同一模型在不同 split 上的结果仍然落在同一个实验目录下。

## 配置中心

配置从四层合并，后面的层覆盖前面的同名字段：

1. `configs/common/base.yaml`：所有任务共享的 batch、设备、优化器和产出根目录默认值。
2. `configs/common/sr.yaml`、`refsr.yaml`、`refsrwkv.yaml`、`refdiffrwkv.yaml`：任务或模型家族的训练默认值。
3. `configs/datasets/sr/*.yaml`、`configs/datasets/refsr/*.yaml`：数据集 id、物理路径、原始尺寸、split 数量和参考图策略。
4. `configs/models/*/*.yaml` 与 `configs/runs/**/*.yaml`：网络结构和一次可复现实验的 scale、patch、损失开关。

例如 `configs/runs/refsrwkv/real_refrssrd_x10.yaml` 只声明数据集、x10 和少数差异；它的通用训练参数来自 `common/refsrwkv.yaml`，网络参数来自 `models/refsr/refsrwkv.yaml`。命令行可以用 `--overrides model.dim=64 train.learning_rate=5e-5` 覆盖任意点路径。

配置字段的归属约定如下：

- `data.*`：数据根目录、split、增强、batch 和键名。
- `model.*`：网络结构、扩散开关、参考分支和 prior 路径。
- `train.*`：优化器、scheduler、精度、checkpoint、early stopping 和设备。
- `loss.*`：L1/SSIM/FFT、扩散、LPIPS、GAN 等目标权重。
- `output.*`：通常不需要手写，运行时由 `runtime.experiments` 自动物化。

## 数据隔离与动态倍率

### SR

`data/sr/<dataset>/<split>/{HR,LR}` 是严格的 HR/LR 配对。`data.sr.dataset.SRPNGDataset` 只读取这两个目录，不接受 `Ref`。因此 SR 数据目录中不保留由旧流程生成的 `Ref/` 副本。

### RefSR

`data/refsr/<dataset>/<split>/{HR,LR,Ref}` 是严格的三元组。`data.refsr.dataset.RefPNGDataset` 只负责这类真实三元组，并强制检查 `Ref`。在 AID 或 UC Merced 这类没有真实参考图的数据集上，RefSRWKV 应使用 `reference_mode: lr_up`；`data.loaders.build_refsr_loaders` 会改用 `SRPNGDataset`，运行时从当前 LR 生成 bicubic 参考，不会修改原始数据。

配置也按该契约分层：`common/refsr.yaml` 和 `common/refsrwkv.yaml` 只放两个模式都可用的默认值；`common/refsr_paired.yaml`、`common/refsrwkv_paired.yaml` 才包含 `augment_ref`、颜色/灰度参考图增强和 `loss.ref_drop_prob`。`lr_up` 不允许出现这些字段，避免真实 Ref 增强配置被静默忽略。`RefDiffRWKV` 当前固定使用 `paired`，训练与采样都必须得到真实 `LR/HR/Ref` 三元组。

`configs/models/refsr/refsrwkv.yaml` 还集中声明了 RefSRWKV 的消融开关：`model.fusion_match.enabled` 关闭时退回 v1 的逐位置 cosine 融合，`model.fusion_match.window` 可设统一奇数窗口或按 `enc1/enc2/enc3/latent/dec3/dec2/dec1` 分阶段设置；`model.fusion_match.conf` 和 `model.fusion_match.quality` 分别关闭匹配熵置信度与质量门控。`model.decoder_refusion` 控制解码器 skip 后的二次参考注入，`model.global_latent_blocks` 取 `0/1/2`，`model.ref_encoder` 取 `shallow/deep`（分别为 HR 域一层/两层 3x3 卷积）。这些字段也可用 `--overrides` 点路径覆盖，默认值保持当前完整模型结构。由于大多数开关会改变参数集合或形状，消融模型应从头训练，或只加载相同配置生成的 checkpoint。

### scale

磁盘上的 LR 是一种存储表示，通常只保留数据集准备时的倍率（当前 AID、UC Merced、HRMS-SCD 为 x4，Real-RefRSSRD 为 x10）。加载器读取 HR 后，按照 `run.scale` 在内存中重采样 LR，再检查 `HR = LR * scale`。因此同一份数据可以尝试 x2、x4 或其他正整数倍率；不会生成默认的 `cache/`，也不会把派生 LR 写回数据目录。若以后确实需要磁盘缓存，应放在被忽略的 `cache/lr/<dataset>/x<scale>/`，而不是提交到 `data/` 或源码目录。

训练 patch 在 LR 网格上采样，再乘 scale 映射到 HR/Ref，保证三个张量像素对齐。验证和测试默认使用原图分辨率，只有显式设置 `data.val_patch_size` 或 `data.test_patch_size` 才裁剪。

## 模型代码如何解耦

### SR registry

`models/sr/registry.py` 只定义模型注册和构造接口。每个新 SR 模型建立自己的目录，例如：

```text
models/sr/rcan/
├── network.py       # 纯 torch 网络
├── adapter.py       # 将配置和 [-1, 1] 项目契约接到 registry
└── __init__.py
```

在 `adapter.py` 中调用 `register_adapter(RCANAdapter())`，再新增一个模型 YAML；训练器、数据加载器、指标和测试脚本不需要复制。

### RefSR 模型家族

RefSR 目前只保留两个模型目录：

- `models/refsr/refsrwkv/`：独立的参考图超分网络，既可以直接训练，也可以作为扩散模型的 SR prior。
- `models/refsr/RefDiffRWKV/`：扩散生成器、参考适配器、语义模块、判别器、采样器和系统封装。

扩散模型不是单独的 `prior/` 或 `diffusion/` 顶层任务。它属于 RefSR，并通过 `model.sr.ckpt_path` 加载 `RefSRWKV` 或其他兼容 SR 网络的权重；`model.sr_fixed: true` 时 prior 冻结，设为 false 时可以联合微调。替换 prior 只需要替换构造器/权重配置，不改变数据契约。

跨模型复用的通用训练/运行时逻辑放在 `engines/`、`runtime/`、`losses/`、`metrics/`；WKV CUDA 源码只放在 `kernels/wkv/`。模型目录可以依赖这些公共模块，但公共模块不能反向 import 某个具体模型，避免循环依赖。

## 训练 engine

`engines/base_trainer.py` 统一处理 AdamW、EMA、梯度裁剪、验证指标、checkpoint 元数据、Plateau/Cosine scheduler 和 Lightning 生命周期。`engines/sr/trainer.py` 与 `engines/refsr/refsrwkv_trainer.py` 只负责解包 batch、前向和损失的差异。

`RefDiffRWKV` 保留 `engines/refsr/refdiff_trainer.py` 这个稳定入口，但其 G/D 交替、手动梯度累积和 AMP 是扩散系统本身的必要协议，因此不强行改写成普通单优化器 `BaseTrainer`。它仍使用相同的实验目录、配置快照和 checkpoint 规则。

## 权重、checkpoint、日志和测试结果

这几个目录的职责不同：

- `weights/pretrained/`：下载的基础模型、迁移来的旧 checkpoint、不会随某次实验自动覆盖的权重。当前旧 `checkpoints/` 会迁移到 `weights/pretrained/legacy/`。
- `weights/exports/`：从实验中明确导出的部署或分享权重。
- `experiments/train/.../checkpoints/`：一次训练运行的 `last.ckpt`、top-k checkpoint 和恢复所需的配置快照。训练产出不放 `weights/experiments`。
- `experiments/train/.../logs/`：TensorBoard event 文件和训练日志；用 `tensorboard --logdir experiments/train` 查看。
- `experiments/test/.../`：推理图片和 `metrics.json`；测试产出不放 `results/`。

标准路径为：

```text
experiments/train/<task>/<model>/<dataset>/x<scale>/<run>/
├── checkpoints/
├── logs/
├── config.json
└── config.yaml

experiments/test/<task>/<model>/<dataset>/x<scale>/<run>/<split>/
├── images/
└── metrics.json
```

`runtime.checkpoint.load_model_weights` 同时支持裸 state dict、Lightning checkpoint、EMA state 和旧模型的外层前缀。新实验仍建议从 `experiments/train/.../checkpoints/last.ckpt` 或 `weights/pretrained/...` 明确指定来源。

## 训练、测试和评估

单图 SR（SwinIR-M x4）：

```bash
python scripts/train/sr.py --config configs/runs/sr/swinir/aid_x4.yaml
python scripts/test/sr.py \
  --config configs/runs/sr/swinir/aid_x4.yaml \
  --checkpoint experiments/train/sr/swinir/aid/x4/aid_x4/checkpoints/last.ckpt \
  --split test
```

RefSRWKV：

```bash
python scripts/train/refsrwkv.py --config configs/runs/refsrwkv/hrms_scd_x4.yaml
python scripts/test/refsr.py \
  --config configs/runs/refsrwkv/hrms_scd_x4.yaml \
  --checkpoint experiments/train/refsr/refsrwkv/hrms_scd/x4/hrms_scd_x4/checkpoints/last.ckpt \
  --split test_easy
```

RefDiffRWKV 的四阶段配置都在 `configs/runs/refdiffrwkv/`。把训练好的 RefSRWKV 权重写入 `model.sr.ckpt_path`，然后运行：

```bash
python scripts/train/refdiffrwkv.py --config configs/runs/refdiffrwkv/stage1.yaml
python scripts/evaluate.py \
  --config configs/runs/refdiffrwkv/stage1.yaml \
  --checkpoint experiments/train/refsr/refdiffrwkv/real_refrssrd/x10/stage1/checkpoints/last.ckpt \
  --split test
```

`scripts/evaluate.py` 是 SR 和 RefSR 共用的评估入口；`evaluation/runner.py` 负责选择 loader、构造模型、保存 PNG 和汇总 PSNR/SSIM。更重的 LPIPS、DISTS、SAM 等离线指标实现放在 `evaluation/eval_pyiqa.py` 和 `evaluation/eval_sewar.py`，不参与训练主循环。

## 添加数据集

1. 按物理数据契约准备 `train/val/test`（真实 RefSR 还可以有 `test_easy/test_hard`）。`reference_mode: lr_up` 的 RefSRWKV 复用 SR 的 `HR/LR` 契约；`paired` 和 RefDiffRWKV 使用 `HR/LR/Ref` 三元组。
2. 将 SR 数据和 `lr_up` RefSRWKV 要复用的数据放到 `data/sr/<id>`；只将真实参考图三元组放到 `data/refsr/<id>`。原始压缩包放到 `data/raw` 或 `data/archives`。
3. 新增 `configs/datasets/{sr,refsr}/<id>.yaml`，填写 `root`、原始尺寸、scale 和 split 统计。
4. 从一个现有 run YAML 复制出新实验，只修改 `dataset.config`、`run.name`、`run.scale` 和 patch/batch 差异。
5. 先用 `--overrides data.max_samples_train=8 data.max_samples_val=2` 做 loader smoke test，再正式训练。

### 聚合多个数据集

训练 loader 既可接收一个数据集根目录，也可接收任务根目录。`data.root: data/sr` 会自动发现下一层中满足 `train/{HR,LR}` 和 `val/{HR,LR}` 契约的所有数据集，可供 SR 或 `lr_up` RefSRWKV 使用；`data.root: data/refsr` 仅用于 `paired` RefSR，额外要求 `Ref/`。新增完整数据集后不需要改 loader 代码。

若只想组合部分数据集，配置中使用 `data.roots`：

```yaml
data:
  roots:
    - data/sr/AID
    - data/sr/UC_Merced
```

聚合训练使用 `ConcatDataset`。训练 crop 会统一到当前 `run.scale` 和 `run.lr_patch`；验证默认保持原图大小，因此混合分辨率时应保持 `data.val_batch_size: 1`。

## 添加模型

SR 模型按 registry/adaptor 约定接入 `models/sr/`。RefSR 新模型必须在 `models/refsr/<ModelName>/` 中自包含，明确 `forward(lr, ref)` 或扩散系统的输入输出值域，并在 `scripts/test` 的构造分支中注册。训练器只读取数据键名和模型接口，不直接 import 另一个模型的私有实现。

## 迁移后的旧目录

旧的 `baselines/`、顶层 `models/RefSRWKV.py`、`models/RefDiffRWKV/`、`models/EnRWKV.py`、`models/cuda/`、旧训练/测试脚本和重复 YAML 已移除。旧数据、checkpoint、日志和结果只做同文件系统移动，不重新编码图片，也不覆盖新目录中已经存在的文件。
