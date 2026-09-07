# RefRWKV 目录结构

本仓库按“任务、数据、模型、运行时产出”拆分。代码目录只放可复用代码，数据、权重、日志和推理结果分别放在自己的生命周期目录中。`SR`（只有 LR/HR）和 `RefSR`（LR/HR/Ref）使用不同的数据契约，加载器不会相互兜底读取。

## 目录总览

```text
RefRWKV/
├── configs/
│   ├── common/                 # 任务公共默认值与 benchmark 比较协议
│   ├── datasets/{sr,refsr}/    # 数据集元信息与路径
│   ├── models/{sr,refsr}/      # 网络结构默认值
│   └── runs/                   # 可直接启动的实验配置
├── data/
│   ├── dataset.py              # 统一 HR/LR/Ref Dataset
│   ├── sr/                     # SR 数据集（HR，可选 LR）
│   ├── refsr/                  # RefSR 数据集（HR、LR，可选 Ref）
│   ├── raw/sr/                 # SR 原始压缩包或解压缓存
│   └── archives/refsr/         # RefSR 原始压缩包
├── models/
│   ├── sr/                     # 单图 SR 模型与 registry
│   │   ├── swinir/             # SwinIR 网络和 adapter
│   │   └── baselines.py        # Bicubic、EDSR、RCAN、HAT、MambaIRv2 compatibility
│   └── refsr/
│       ├── refsrwkv/           # 参考超分 RWKV 模型
│       ├── baselines.py        # TTSR、MASA-SR、DATSR compatibility
│       └── RefDiffRWKV/        # 参考超分扩散模型及其 G/D 组件
├── engines/
│   ├── base_trainer.py         # 训练公共生命周期
│   ├── sr/trainer.py           # SR 的 train/eval step
│   └── refsr/                  # direct RefSR、RefSRWKV 与 RefDiffRWKV engine
├── kernels/wkv/                # WKV CUDA 源码，模型只通过统一接口调用
├── losses/                     # 可组合损失
├── metrics/                    # 指标实现
├── evaluation/                # 推理、指标汇总和结果写盘
├── runtime/                    # 配置、checkpoint、实验路径、EMA 等运行时工具
├── scripts/
│   ├── train/                  # sr.py、refsr.py、refsrwkv.py、refdiffrwkv.py
│   ├── test/                   # SR/RefSR 推理入口
│   ├── shortcuts/              # 按模型/数据集/任务/倍率命名的快捷训练脚本
│   ├── prepare/                # 数据准备脚本
│   └── evaluate.py             # 统一评估入口
├── experiments/
│   ├── train/                  # 每次训练的 checkpoint、config、TensorBoard 日志
│   └── test/                   # 每次测试的图片和 metrics.json
├── weights/
│   ├── pretrained/              # 外部或下载的预训练权重
│   └── exports/                 # 明确导出的部署权重
├── tests/                      # 轻量 smoke test 与回归测试
├── environments/               # 可选官方基线的独立 Conda 环境定义
└── docs/                       # 架构和数据说明
```

`test` 是测试产出的推荐名称；`test_easy` 和 `test_hard` 是数据集内部的 split，而不是另一个顶层目录。同一模型在不同 split 上的结果统一落在同一个实验目录下。

## 配置中心

配置从四层合并，后面的层覆盖前面的同名字段：

1. `configs/common/base.yaml`：所有任务共享的 batch、设备、优化器和产出根目录默认值。
2. `configs/common/sr.yaml`、`refsr.yaml`、`refsrwkv.yaml`、`refdiffrwkv.yaml`：任务或模型家族的训练默认值。
3. `configs/datasets/sr/*.yaml`、`configs/datasets/refsr/*.yaml`：数据集 id、物理路径、原始尺寸、split 数量和参考图策略。
4. `configs/models/*/*.yaml` 与 `configs/runs/**/*.yaml`：网络结构和一次可复现实验的 scale、patch、损失开关。

HRMS-SCD x4 的完整对比 run 在上述 base 列表的最后再引入
`configs/common/benchmark.yaml`，以覆盖模型家族自己的训练默认值。它固定
`1e-4`、Plateau、每 epoch 验证、纯 L1、`max_epochs: -1`、`max_steps: 50000`
和禁用 early stopping；完整范围见[对比基线表](models/baselines.md)。

例如 `configs/runs/refsrwkv/real_refrssrd_x10.yaml` 只声明数据集、x10 和少数差异；它的通用训练参数来自 `common/refsrwkv.yaml`，网络参数来自 `models/refsr/refsrwkv.yaml`。命令行可以用 `--overrides model.dim=64 train.learning_rate=5e-5` 覆盖任意点路径。

配置字段的归属约定如下：

- `data.*`：数据根目录、split、增强、batch 和键名。
- `model.*`：网络结构、扩散开关、参考分支和 prior 路径。
- `train.*`：优化器、scheduler、精度、checkpoint、early stopping 和设备。
- `loss.*`：L1/SSIM/FFT、扩散、LPIPS、GAN 等目标权重。
- `output.*`：通常不需要手写，运行时由 `runtime.experiments` 自动物化。

## 数据隔离与动态倍率

### Unified Dataset

`data.dataset.SuperResolutionDataset` 是 SR 与 RefSR 共用的 PNG Dataset。它按 `return_items=("lr", "hr")` 或 `("lr", "hr", "ref")` 返回字段；`data.sr.dataset.SRPNGDataset` 和 `data.refsr.dataset.RefPNGDataset` 提供按任务命名的调用入口，内部使用同一 Dataset 实现。

LR 来源由三个字段明确描述：

- `data.lr_source=auto|stored|from_hr`：自动优先读取匹配的磁盘 LR、强制读取磁盘 LR，或始终从 HR bicubic 生成。
- `data.lr_native_scale`：磁盘 LR 的原生倍率。
- `data.lr_provenance=bicubic|sensor`：AID、UC Merced、HRMS-SCD 是 HR bicubic 下采样；Real-RefRSSRD 是 Sentinel-2 实测 LR。

`bicubic` 数据允许在其他倍率从 HR 重新生成 LR；`sensor` 数据禁止重采样，要求请求倍率等于原生倍率并存在磁盘 LR。RefSRWKV 的 `reference_mode: lr_up` 只返回 `lr/hr`，由 trainer 从 LR 生成 bicubic 自参考；`paired` 才读取 `Ref`。TTSR、MASA-SR、DATSR 和 RefDiffRWKV 被配置校验强制为 `paired`，不会把 bicubic(LR) 误作跨时相参考图。因此 SwinIR 在 RefSR 目录上训练时可以复用同一个 Dataset 并忽略 `Ref`。

配置也按该契约分层：`common/refsr.yaml` 和 `common/refsrwkv.yaml` 只放两个模式都可用的默认值；`common/refsr_paired.yaml`、`common/refsrwkv_paired.yaml` 才包含 `augment_ref`、颜色/灰度参考图增强和 `loss.ref_drop_prob`。`lr_up` 不允许出现这些字段，避免真实 Ref 增强配置被静默忽略。`RefDiffRWKV` 固定使用 `paired`，训练与采样都必须得到真实 `LR/HR/Ref` 三元组。

`configs/models/refsr/refsrwkv.yaml` 还集中声明了 RefSRWKV 的消融开关：`model.fusion_match.enabled` 关闭时使用逐位置 cosine 融合路径，`model.fusion_match.window` 可设统一奇数窗口或按 `enc1/enc2/enc3/latent/dec3/dec2/dec1` 分阶段设置；`model.fusion_match.conf` 和 `model.fusion_match.quality` 分别关闭匹配熵置信度与质量门控。`model.decoder_refusion` 控制解码器 skip 后的二次参考注入，`model.global_latent_blocks` 取 `0/1/2`，`model.ref_encoder` 取 `shallow/deep`（分别为 HR 域一层/两层 3x3 卷积）。这些字段也可用 `--overrides` 点路径覆盖，默认值启用完整模型结构。由于大多数开关会改变参数集合或形状，消融模型应从头训练，或只加载相同配置生成的 checkpoint。

### scale

磁盘上的 LR 是一种存储表示，通常只保留数据集准备时的倍率（AID、UC Merced、HRMS-SCD 为 x4，Real-RefRSSRD 为 x10）。对 bicubic 数据，加载器读取 HR 后，按照 `run.scale` 在内存中重新生成 LR，再检查 `HR = LR * scale`；不会写回 `data/`。对 sensor 数据，倍率必须保持原生值，避免把实测 LR 错误地 resize 成另一个物理观测。

训练 patch 在 LR 网格上采样，再乘 scale 映射到 HR/Ref，保证三个张量像素对齐。验证和测试默认使用原图分辨率，只有显式设置 `data.val_patch_size` 或 `data.test_patch_size` 才裁剪。

## 模型代码如何解耦

### SR registry

`models/sr/registry.py` 只定义模型注册和构造接口。现有的 SwinIR 使用独立目录；
轻量、无额外依赖的比较模型集中在 `baselines.py` 与 `baseline_adapters.py`。后续
每个 SR 模型也可以建立自己的目录，例如：

```text
models/sr/rcan/
├── network.py       # 纯 torch 网络
├── adapter.py       # 将配置和 [-1, 1] 项目契约接到 registry
└── __init__.py
```

在 `adapter.py` 中调用 `register_adapter(RCANAdapter())`，再增加一个模型 YAML；训练器、数据加载器、指标和测试脚本不需要复制。

### RefSR 模型家族

RefSR 模型目录包括：

- `models/refsr/refsrwkv/`：独立的参考图超分网络，既可以直接训练，也可以作为扩散模型的 SR prior。
- `models/refsr/baselines.py`：TTSR、MASA-SR、DATSR 的 `forward(lr, ref) -> sr` direct RefSR compatibility 实现，统一使用 RGB `[-1, 1]` 张量。
- `models/refsr/RefDiffRWKV/`：扩散生成器、参考适配器、语义模块、判别器、采样器和系统封装。

direct RefSR 通过 `models/refsr/registry.py` 注册，并由
`engines/refsr/trainer.py` 与 `scripts/train/refsr.py` 共享训练循环。官方代码或
官方 checkpoint 需要独立 bridge 环境，不能与 `native_compatibility` checkpoint
互换；环境和可比性边界见[对比基线表](models/baselines.md)。

扩散模型属于 RefSR，并通过 `model.sr.ckpt_path` 加载 `RefSRWKV` 或其他可作为先验的 SR 网络权重；`model.sr_fixed: true` 时 prior 冻结，设为 false 时可以联合微调。替换 prior 只需要替换构造器和权重配置，不改变数据契约。

跨模型复用的通用训练/运行时逻辑放在 `engines/`、`runtime/`、`losses/`、`metrics/`；WKV CUDA 源码只放在 `kernels/wkv/`。模型目录可以依赖这些公共模块，但公共模块不能反向 import 某个具体模型，避免循环依赖。

## 训练 engine

`engines/base_trainer.py` 统一处理 AdamW、EMA、梯度裁剪、验证指标、checkpoint 元数据、Plateau/Cosine scheduler 和 Lightning 生命周期。`engines/sr/trainer.py`、`engines/refsr/trainer.py` 与 `engines/refsr/refsrwkv_trainer.py` 只负责解包 batch、前向和损失的差异；其中 direct RefSR trainer 适用于任意 `forward(lr, ref)` 模型。

`engines/refsr/refdiff_trainer.py` 负责 RefDiffRWKV 的 G/D 交替、手动梯度累积和 AMP；这些步骤是扩散系统本身的训练协议。它使用与其他任务相同的实验目录、配置快照和 checkpoint 规则。

## 权重、checkpoint、日志和测试结果

这几个目录的职责不同：

- `weights/pretrained/`：下载的基础模型和可复用 checkpoint，不会随某次实验自动覆盖。
- `weights/exports/`：从实验中明确导出的部署或分享权重。
- `experiments/train/.../checkpoints/`：保存一次训练运行的 `last.ckpt`、top-k checkpoint 和恢复所需的配置快照。
- `experiments/train/.../logs/`：TensorBoard event 文件和训练日志；用 `tensorboard --logdir experiments/train` 查看。
- `experiments/test/.../`：保存推理图片和 `metrics.json`；后者也记录 `implementation`、`variant` 和 `reference_mode`，供结果汇总时区分 native 与官方 bridge。

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

`runtime.checkpoint.load_model_weights` 同时支持裸 state dict、Lightning checkpoint、EMA state 和带参数前缀的 checkpoint。运行时建议从 `experiments/train/.../checkpoints/last.ckpt` 或 `weights/pretrained/...` 明确指定来源。

## 训练、测试和评估

分层 run 配置可以先展开成单个可编辑 YAML：

```bash
python scripts/render_config.py \
  --config configs/runs/refsrwkv/hrms_scd_trefsr_x4.yaml
```

省略 `--output` 时，完整文件写入对应的 `experiments/train/.../config.yaml`。编辑后直接将该文件传给训练或评估入口即可；已有文件默认不会被覆盖，确认重新生成时加 `--force`。如果再次传入对应的 `configs/runs/...`，加载器会优先使用这个快照；直接传入完整 YAML 则始终使用该文件。

单图 SR（SwinIR-M x4）：

```bash
python scripts/train/sr.py --config configs/runs/sr/swinir/aid_x4.yaml
python scripts/test/sr.py \
  --config configs/runs/sr/swinir/aid_x4.yaml \
  --checkpoint experiments/train/sr/swinir/aid/x4/aid_x4/checkpoints/last.ckpt \
  --split test
```

其他单图比较模型共用同一入口，例如：

```bash
python scripts/train/sr.py --config configs/runs/sr/rcan/hrms_scd_x4.yaml
python scripts/test/sr.py \
  --config configs/runs/sr/bicubic/hrms_scd_x4.yaml \
  --split test_easy
```

RefSRWKV：

```bash
python scripts/train/refsrwkv.py --config configs/runs/refsrwkv/hrms_scd_x4.yaml
python scripts/test/refsr.py \
  --config configs/runs/refsrwkv/hrms_scd_x4.yaml \
  --checkpoint experiments/train/refsr/refsrwkv/hrms_scd/x4/hrms_scd_x4/checkpoints/last.ckpt \
  --split test_easy
```

TTSR、MASA-SR、DATSR 共享 direct RefSR 入口，例如：

```bash
python scripts/train/refsr.py --config configs/runs/refsr/datsr/hrms_scd_x4.yaml
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

1. 按物理数据契约准备 `train/val/test`（真实 RefSR 还可以有 `test_easy/test_hard`）。最小目录是 `HR/`；`LR/` 对 bicubic 数据可省略并由 HR 在线生成，sensor 数据必须存在；`Ref/` 只在 paired RefSR 中需要。
2. 将 SR 数据和 `lr_up` RefSRWKV 要复用的数据放到 `data/sr/<id>`，真实参考图三元组也可以放到 `data/refsr/<id>`。SwinIR 是否使用 `Ref` 由 loader 的任务模式决定，而不是目录位置决定。原始压缩包放到 `data/raw` 或 `data/archives`。
3. 添加 `configs/datasets/{sr,refsr}/<id>.yaml`，填写 `root`、原始尺寸、scale、split 统计，以及 `lr: {native_scale: N, provenance: bicubic|sensor, source: auto|stored|from_hr}`。
4. 参考一个 run YAML 创建实验配置，只修改 `dataset.config`、`run.name`、`run.scale` 和 patch/batch 差异。
5. 先用 `--overrides data.max_samples_train=8 data.max_samples_val=2` 做 loader smoke test，再正式训练。

### 聚合多个数据集

训练 loader 既可接收一个数据集根目录，也可接收任务根目录。`data.root: data/sr` 会自动发现下一层中满足 `train/{HR,LR}` 和 `val/{HR,LR}` 契约的所有数据集，可供 SR 或 `lr_up` RefSRWKV 使用；`data.root: data/refsr` 仅用于 `paired` RefSR，额外要求 `Ref/`。数据集满足契约后不需要改 loader 代码。

若只想组合部分数据集，配置中使用 `data.roots`：

```yaml
data:
  roots:
    - data/sr/AID
    - data/sr/UC_Merced
```

聚合训练使用 `ConcatDataset`。训练 crop 会统一到当前 `run.scale` 和 `run.lr_patch`；验证默认保持原图大小，因此混合分辨率时应保持 `data.val_batch_size: 1`。

## 添加模型

SR 模型按 registry/adaptor 约定接入 `models/sr/`。direct RefSR 模型实现
`forward(lr, ref)` 并注册 adapter 后，可直接复用 `scripts/train/refsr.py` 与统一
评估入口；扩散模型才需要专用 engine。训练器只读取数据键名和模型接口，不直接
import 另一个模型的私有实现。
