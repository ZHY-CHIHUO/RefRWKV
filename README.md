# RefRWKV

RefRWKV 是一个参考图超分辨率（RefSR）和单图超分辨率（SR）实验仓库。代码、数据、权重、训练运行和测试结果按生命周期分开；添加模型只需要加入自己的模型目录、配置和入口注册，不需要复制数据加载或评估逻辑。

各一级目录的具体说明放在对应的 `README.md`：[`configs/`](configs/README.md)、[`data/`](data/README.md)、[`models/`](models/README.md)、[`scripts/`](scripts/README.md)、[`runtime/`](runtime/README.md)、[`experiments/`](experiments/README.md) 等。本文只保留项目总览和常用入口。

## 目录

```text
RefRWKV/
├── configs/                       # 分层 YAML 配置
│   ├── common/                   # 全局、任务默认值与 benchmark 比较协议
│   ├── datasets/sr/              # SR 数据集配置及 LR 来源元数据
│   ├── datasets/refsr/           # RefSR 数据集配置及 LR 来源元数据
│   ├── models/sr/                # SR 网络默认参数
│   ├── models/refsr/              # RefSR 网络默认参数
│   └── runs/                     # 可直接运行的实验配置
├── data/
│   ├── dataset.py                # 统一 HR/LR/Ref Dataset
│   ├── sr/                       # SR 数据：<dataset>/<split>/{HR[,LR]}
│   ├── refsr/                    # RefSR 数据：图像、Wuhan TIFF、PanCollection H5 加载器
│   ├── raw/                      # 原始压缩包和解压内容
│   └── archives/                 # RefSR 多卷压缩包等归档
├── models/
│   ├── sr/                       # SR registry、SwinIR 和对比模型实现
│   └── refsr/
│       ├── refsrwkv/             # 参考图超分 RWKV 模型
│       ├── baselines.py           # TTSR、MASA-SR、DATSR compatibility 基线
│       └── RefDiffRWKV/          # 参考图扩散模型、适配器、G/D、采样器
├── engines/                      # 训练生命周期
│   ├── base_trainer.py           # optimizer、EMA、验证、checkpoint 等公共逻辑
│   ├── sr/trainer.py             # SR 差异化 train/eval step
│   └── refsr/                    # RefSRWKV 与 RefDiffRWKV engine
├── runtime/                      # 配置、权重加载、实验路径和通用运行时工具
├── evaluation/                   # 推理和指标写盘
├── losses/                       # 可组合损失
├── metrics/                      # PSNR、SSIM 等指标
├── kernels/wkv/                  # WKV CUDA 源码
├── scripts/
│   ├── train/                    # sr.py、refsr.py、refsrwkv.py、refdiffrwkv.py
│   ├── test/                     # sr.py、refsr.py
│   ├── shortcuts/
│   │   ├── train/                # 训练快捷脚本
│   │   └── test/                 # 测试快捷脚本
│   ├── prepare/                  # 数据准备脚本
│   └── evaluate.py               # 统一评估入口
├── experiments/
│   ├── train/                    # 每个 run 的 checkpoint、配置快照和日志
│   └── test/                     # 每个 split 的图片和 metrics.json
├── weights/
│   ├── pretrained/               # 外部或下载的权重
│   └── exports/                  # 明确导出的部署权重
├── tests/                        # smoke/regression tests
├── environments/                 # 可选官方基线的独立 Conda 环境定义
└── README.md                     # 项目总览；各目录内另有 README.md
```

`RefSRWKV` 和 `RefDiffRWKV` 都属于 `models/refsr/`。扩散模型可以通过 `model.sr.ckpt_path` 使用 RefSRWKV 或其他可作为先验的 SR 网络，扩散相关代码和配置均位于 RefSR 的模型与运行目录中。

## 环境

先按本机 CUDA 版本从 PyTorch 官网安装匹配的 `torch`/`torchvision`，再安装 SR 和 RefSRWKV 的核心依赖：

```bash
python -m pip install -r requirements.txt
```

RefDiffRWKV、离线 LPIPS/DISTS/SAM 指标额外安装：

```bash
python -m pip install -r requirements-refdiff.txt
```

Stage 4 还需要 `vision_aided_loss`，其上游安装命令写在 `requirements-refdiff.txt`。WKV CUDA 后端需要本机 CUDA toolkit/NVCC；模型导入和 `--help` 不会主动编译扩展。

EDSR、RCAN、HAT、MambaIRv2、TTSR、MASA-SR、DATSR 的可训练 compatibility
基线已包含在主环境；官方旧代码或编译依赖不要混装进 `rwkv7`。环境 YAML、风险
和结果可比性边界见 [环境说明](environments/README.md) 与
完整基线协议和模型说明见 [`models/README.md`](models/README.md)。

## 配置

运行配置从四层合并：

1. `configs/common/base.yaml`：设备、batch、优化器和输出根目录默认值。
2. `configs/common/*.yaml`：SR、RefSRWKV、RefDiffRWKV 家族默认值。
3. `configs/datasets/{sr,refsr}/*.yaml`：数据集路径、尺寸和 split。
4. `configs/models/` 与 `configs/runs/`：网络参数和一次实验的倍率、patch、损失开关。

命令行可以用 `--overrides model.dim=64 train.learning_rate=5e-5` 覆盖点路径字段。bicubic 数据会在运行时从 HR 按目标倍率生成 LR；sensor 数据保持原生观测倍率，不默认生成 `cache/`，也不会修改 `data/`。

`data.root` 可以是单个数据集（如 `data/sr/AID`），也可以是任务目录（如 `data/sr`）。后者会自动聚合下一层数据集；也可用 `data.roots=[...]` 明确指定组合。常规 SR/RefSR loader 使用 `data.dataset.SuperResolutionDataset`，由 `return_items` 决定返回 `lr`、`hr`、`ref`；Wuhan 的 temporal-pair GeoTIFF 使用专用 `WuhanSTFDataset`；PanCollection 的 `train_wv3.h5` 等文件使用通用 `PanCollectionH5Dataset`，由 YAML 将 `ms/pan/gt` 映射为 `lr/ref/hr`。`data.lr_source` 支持 `auto`、`stored`、`from_hr`：AID、UC Merced 和 `data/refsr/HRMS_SCD` 的 LR 都是 HR 经 bicubic 下采样 x4 生成（可在 HR-only 目录上在线生成），Real-RefRSSRD 的 LR 是实测 Sentinel-2，配置为 `stored` 且只能使用原生 x10。RefSRWKV 使用 `data.reference_mode: lr_up` 时返回 `lr/hr`，trainer 从 LR 动态生成 bicubic 自参考；只有 `paired` 才返回并读取 `Ref`。

真实参考图专属字段只放在 `configs/common/refsr_paired.yaml` 和 `configs/common/refsrwkv_paired.yaml`：`augment_ref`、`ref_aug_strengths`、`ref_aug_probs`、`ref_gray_prob`、`loss.ref_drop_prob`。`lr_up` 运行配置不能设置它们，配置校验会直接报错，避免参数看似启用但实际无效。`RefDiffRWKV` 使用 `paired`，因为它的训练和采样都需要真实 `Ref`。

RefSRWKV 的消融开关集中在 `configs/models/refsr/refsrwkv.yaml`：`model.fusion_match.enabled/window/conf/quality`、`model.decoder_refusion`、`model.global_latent_blocks` 和 `model.ref_encoder`。例如 `--overrides model.fusion_match.enabled=false model.global_latent_blocks=0 model.ref_encoder=shallow` 可复现实验变体；默认值对应完整模型。结构型开关通常会改变参数集合或形状，每个变体应从头训练，或只加载同一变体产生的 checkpoint。

## 训练

### Wuhan 四通道时空数据

Wuhan 的 `L`/`G` TIFF 已在文件中对齐为同一 1000×1000 网格，RefSRWKV 使用
`L_t2 + G_t1 -> G_t2`，所以网络配置是 `scale=1`、`inp/ref/out_channels=4`；
30/8=3.75 只用于 ERGAS。数据已整理到 `data/refsr/Wuhan-dataset`，训练/验证/测试
分别为 8/2/5 个完整时相对，训练时四幅影像同步裁剪。运行：

```bash
python scripts/train/refsrwkv.py --config configs/runs/refsrwkv/wuhan.yaml
python scripts/test/run.py --config configs/test/test.yaml \
  --checkpoint experiments/train/refsr/refsrwkv/wuhan/x1/wuhan/checkpoints/last.ckpt \
  --split test
```

Wuhan 评估结果会额外写入 RMSE、UIQI、PSNR、SAM（弧度/角度）和 ERGAS。
验证/测试会以重叠 128×128 tiles 拼接完整网格，避免全图局部匹配的显存峰值。
若从同构的 3 通道 HRMS-SCD RefSRWKV checkpoint 微调，可在训练命令后加
`--load-weights <checkpoint>`；配置会安全适配输入/参考边界层的第四通道，其余
倍率相关层保持 Wuhan 新初始化。

### 完整配置

每次训练都会把合并后的完整配置保存到该实验目录的 `config.yaml`。快捷脚本再次执行同一个 `configs/runs/...` 时，会优先读取这个完整配置，因此手工修改会继续生效。也可以在正式训练前先展开一份配置，手工修改后直接运行：

```bash
python scripts/render_config.py \
  --config configs/runs/refsrwkv/hrms_scd_trefsr_spectral_detail_x4.yaml

# 编辑 experiments/train/refsr/refsrwkv/hrms_scd/x4/hrms_scd_trefsr_spectral_detail_x4/config.yaml
python scripts/train/refsrwkv.py \
  --config experiments/train/refsr/refsrwkv/hrms_scd/x4/hrms_scd_trefsr_spectral_detail_x4/config.yaml
```

如果目标完整配置已经存在，工具默认拒绝覆盖，以免丢失手工修改；确认需要重新展开时使用 `--force`。完整训练 YAML 已经包含数据、模型、训练和输出字段，不再依赖 `base`，可以直接作为训练入口的 `--config` 参数。测试入口使用 `configs/test/test.yaml`，模型和数据配置从 checkpoint 内嵌的训练 YAML 读取。

SR（SwinIR-M，AID x4）：

```bash
python scripts/train/sr.py \
  --config configs/runs/sr/swinir/aid_x4.yaml
```

HRMS-SCD 的三个对照训练建议按下面顺序执行。三次默认都从头训练，不互相依赖；这样比较更公平。

1. SwinIR-SR（不使用参考图）：

```bash
python scripts/train/sr.py \
  --config configs/runs/sr/swinir/hrms_scd_x4.yaml
```

2. RefSRWKV-SR（`lr_up` 自参考，不读取 `Ref/`）：

```bash
python scripts/train/refsrwkv.py \
  --config configs/runs/refsrwkv/hrms_scd_sr_x4.yaml
```

3. RefSRWKV-TRefSR（`paired` 跨时相参考图）：

```bash
python scripts/train/refsrwkv.py \
  --config configs/runs/refsrwkv/hrms_scd_trefsr_spectral_detail_x4.yaml
```

PanCollection WV3 全色多光谱融合（8 通道 MS LR + 1 通道 PAN Ref）：

```bash
bash scripts/shortcuts/train/refsrwkv_pancollection_wv3.sh
```

完整对比集合（Bicubic、EDSR、RCAN、SwinIR、HAT、MambaIRv2、RefSRWKV、TTSR、
MASA-SR、DATSR）的配置、快捷入口、参数量和复现边界见
[`models/README.md`](models/README.md)。其中 Bicubic 只评估，其余
模型统一使用 `1e-4`、plateau、每 epoch 验证、L1、`max_epochs: -1`、
`max_steps: 50000` 和无 early stopping。

RefDiffRWKV：

```bash
python scripts/train/refdiffrwkv.py \
  --config configs/runs/refdiffrwkv/stage1.yaml
```

训练目录统一为：

```text
experiments/train/<task>/<model>/<dataset>/x<scale>/<run>/
├── checkpoints/                  # last.ckpt、top-k checkpoint
├── logs/                         # TensorBoard event 和训练日志
├── config.json                   # 本次 materialized 配置
└── config.yaml                   # 可读配置快照
```

查看所有训练日志：

```bash
tensorboard --logdir experiments/train
```

## 测试与评估

测试使用独立 YAML。模型、数据集和架构设置直接从 checkpoint 内嵌的最终
`trainer_config` 读取，不会重新加载 `configs/runs` 中的训练配置：

```bash
python scripts/test/run.py \
  --config configs/test/test.yaml \
  --checkpoint experiments/train/sr/swinir/aid/x4/aid_x4/checkpoints/last.ckpt
```

`configs/test/test.yaml` 只控制 split、指标、是否保存图片、设备、batch size、
采样步数和输出路径。命令行参数可以临时覆盖这些设置：

```bash
python scripts/test/run.py \
  --config configs/test/test.yaml \
  --checkpoint experiments/train/refsr/refsrwkv/hrms_scd/x4/hrms_scd_trefsr_x4/checkpoints/last.ckpt \
  --metrics psnr ssim rmse --no-save-images
```

Bicubic 没有 checkpoint，可改用它自己的完整配置提供数据和倍率元信息：

```bash
python scripts/test/run.py \
  --config configs/test/test.yaml \
  --training-config configs/runs/sr/bicubic/hrms_scd_x4.yaml
```

测试输出为：

```text
experiments/test/<task>/<model>/<dataset>/x<scale>/<run>/<split>/
├── images/                        # 1/3 通道为 PNG；其他通道为 float32 TIFF（无 tifffile 时为 NPY）
└── metrics.json                   # PSNR/SSIM；Wuhan 另含 RMSE/UIQI/SAM/ERGAS
```

测试根目录还会保存本次实际使用的 `test_easy.yaml`、`test_hard.yaml`：

```text
experiments/test/<task>/<model>/<dataset>/x<scale>/<run>/
├── test_easy.yaml                 # 可编辑；下一次 test_easy 优先读取
├── test_hard.yaml                 # 可编辑；下一次 test_hard 优先读取
├── test_easy/metrics.json
└── test_hard/metrics.json
```

首次运行时由 `configs/test/test.yaml` 生成这两个文件；生成后直接编辑输出目录里的
对应 YAML 即可生效。只有需要给新实验提供默认策略时，才修改
`configs/test/test.yaml`。

## 权重和日志放置规则

- 外部预训练模型或可复用 checkpoint 放 `weights/pretrained/`。
- 训练过程产生的 `last.ckpt`、最佳 checkpoint 和 TensorBoard 日志放 `experiments/train/.../`，不会放在 `weights/` 根目录。
- 明确导出的部署权重放 `weights/exports/`。
- 推理图片和指标统一放 `experiments/test/.../`。

`runtime.checkpoint.load_model_weights` 支持裸 `state_dict`、Lightning checkpoint、EMA state 以及带参数前缀的 checkpoint；运行时建议显式指定 `--load-weights` 或 `model.sr.ckpt_path`。

## 数据准备

SR 数据必须满足：

```text
data/sr/<dataset>/<split>/{HR,LR}/<same-name>.png
```

RefSR 数据必须满足：

```text
data/refsr/<dataset>/<split>/{HR,LR,Ref}/<same-name>.png
```

AID、UC Merced 的合成 SR 数据可由以下脚本生成：

```bash
python scripts/prepare/remote_sensing.py \
  --dataset aid \
  --source-dir data/raw/sr/aid_extracted/AID \
  --output-dir data/sr/AID
```

将 `--dataset aid` 换成 `ucmerced` 并指定对应原始目录即可。数据说明分别位于 `data/sr/AID/介绍.md`、`data/sr/UC_Merced/介绍.md`、`data/refsr/HRMS_SCD/RefSR-HRMS.md` 和 `data/refsr/Real-RefRSSRD/Real-RefRSSRD.md`。

## 添加模型

添加 SR 模型时，在 `models/sr/<model>/` 放网络和 adapter，并在 adapter 中调用 `register_adapter(...)`；增加一个 `configs/models/sr/<model>.yaml` 和 `configs/runs/sr/<model>/...yaml` 即可复用现有数据、训练和评估流程。

添加直接 RefSR 模型时，实现 `forward(lr, ref) -> sr` 并注册 adapter；它可复用
`engines/refsr/trainer.py` 和 `scripts/train/refsr.py`。只有特殊训练机制（如扩散）
才需要专用 engine。公共 checkpoint、EMA、TensorBoard、scheduler 和 early stopping
逻辑放在 `engines/base_trainer.py`。

模型目录不能直接依赖另一个具体模型目录的私有实现；跨模型复用的训练和运行时能力放在 `engines/`、`runtime/`、`losses/`、`metrics/`，WKV CUDA 代码放 `kernels/wkv/`。

## 验证

```bash
python3 -m compileall -q data engines models runtime scripts evaluation metrics losses tests
python tests/smoke_native_geometry.py --only x4
```

第一个命令不需要导入完整训练依赖即可检查语法；第二个命令需要 CUDA、编译器和 WKV 扩展。
