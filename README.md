# RefRWKV

RefRWKV 是一个参考图超分辨率（RefSR）和单图超分辨率（SR）实验仓库。代码、数据、权重、训练运行和测试结果按生命周期分开；新增模型只需要加入自己的模型目录、配置和入口注册，不需要复制数据加载或评估逻辑。

完整的目录约定、数据契约、迁移说明和扩展规则见 [docs/architecture.md](docs/architecture.md)。

## 目录

```text
RefRWKV/
├── configs/                       # 分层 YAML 配置
│   ├── common/                   # 全局、SR、RefSR、RefSRWKV、RefDiffRWKV 默认值
│   ├── datasets/sr/              # 只有 HR/LR 的 SR 数据集配置
│   ├── datasets/refsr/           # HR/LR/Ref 数据集配置
│   ├── models/sr/                # SR 网络默认参数
│   ├── models/refsr/              # RefSR 网络默认参数
│   └── runs/                     # 可直接运行的实验配置
├── data/
│   ├── sr/                       # SR 数据：<dataset>/<split>/{HR,LR}
│   ├── refsr/                    # RefSR 数据：<dataset>/<split>/{HR,LR,Ref}
│   ├── raw/                      # 原始压缩包和解压内容
│   └── archives/                 # RefSR 多卷压缩包等归档
├── models/
│   ├── sr/                       # SR registry 和模型实现（当前 SwinIR）
│   └── refsr/
│       ├── refsrwkv/             # 参考图超分 RWKV 模型
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
│   ├── train/                    # sr.py、refsrwkv.py、refdiffrwkv.py
│   ├── test/                     # sr.py、refsr.py
│   ├── prepare/                  # 数据准备脚本
│   └── evaluate.py               # 统一评估入口
├── experiments/
│   ├── train/                    # 每个 run 的 checkpoint、配置快照和日志
│   └── test/                     # 每个 split 的图片和 metrics.json
├── weights/
│   ├── pretrained/               # 外部或迁移来的权重
│   └── exports/                  # 明确导出的部署权重
├── tests/                        # smoke/regression tests
└── docs/                         # 架构和数据说明
```

`RefSRWKV` 和 `RefDiffRWKV` 都属于 `models/refsr/`。扩散模型可以通过 `model.sr.ckpt_path` 使用 RefSRWKV 或其他兼容 SR 网络作为先验；仓库不再设置 `prior/` 或 `diffusion/` 顶层目录。

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

## 配置

运行配置从四层合并：

1. `configs/common/base.yaml`：设备、batch、优化器和输出根目录默认值。
2. `configs/common/*.yaml`：SR、RefSRWKV、RefDiffRWKV 家族默认值。
3. `configs/datasets/{sr,refsr}/*.yaml`：数据集路径、尺寸和 split。
4. `configs/models/` 与 `configs/runs/`：网络参数和一次实验的倍率、patch、损失开关。

命令行可以用 `--overrides model.dim=64 train.learning_rate=5e-5` 覆盖点路径字段。倍率在运行时从磁盘 LR 表示重采样，不默认生成 `cache/`，也不会修改 `data/`。

`data.root` 可以是单个数据集（如 `data/sr/AID`），也可以是任务目录（如 `data/sr`）。后者会自动聚合下一层所有完整数据集；也可用 `data.roots=[...]` 明确指定组合。SR 只识别 `HR/LR`，RefSR 才识别 `HR/LR/Ref`。

## 训练

SR（SwinIR-M，AID x4）：

```bash
python scripts/train/sr.py \
  --config configs/runs/sr/swinir/aid_x4.yaml
```

RefSRWKV：

```bash
python scripts/train/refsrwkv.py \
  --config configs/runs/refsrwkv/hrms_scd_x4.yaml
```

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

SR：

```bash
python scripts/test/sr.py \
  --config configs/runs/sr/swinir/aid_x4.yaml \
  --checkpoint experiments/train/sr/swinir/aid/x4/aid_x4/checkpoints/last.ckpt \
  --split test
```

RefSRWKV 或 RefDiffRWKV：

```bash
python scripts/test/refsr.py \
  --config configs/runs/refsrwkv/hrms_scd_x4.yaml \
  --checkpoint experiments/train/refsr/refsrwkv/hrms_scd/x4/hrms_scd_x4/checkpoints/last.ckpt \
  --split test
```

也可以使用统一入口 `python scripts/evaluate.py ...`。测试输出为：

```text
experiments/test/<task>/<model>/<dataset>/x<scale>/<run>/<split>/
├── images/                        # 预测 PNG
└── metrics.json                   # PSNR/SSIM 和样本信息
```

`test` 是推荐的测试目录名；RefSR-HRMS 的 `test_easy`、`test_hard` 是数据集 split，仍然写在同一个实验目录下面。

## 权重和日志放置规则

- 外部预训练模型或迁移的旧 checkpoint 放 `weights/pretrained/`。
- 训练过程产生的 `last.ckpt`、最佳 checkpoint 和 TensorBoard 日志放 `experiments/train/.../`，不会放在 `weights/` 根目录。
- 明确导出的部署权重放 `weights/exports/`。
- 推理图片和指标只放 `experiments/test/.../`，不再使用 `results/`。

`runtime.checkpoint.load_model_weights` 兼容裸 `state_dict`、Lightning checkpoint、EMA state 和旧 checkpoint 外层前缀，因此可以用迁移后的权重热加载；新实验建议显式指定 `--load-weights` 或 `model.sr.ckpt_path`。

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

新增 SR 模型时，在 `models/sr/<model>/` 放网络和 adapter，并在 adapter 中调用 `register_adapter(...)`；新增一个 `configs/models/sr/<model>.yaml` 和 `configs/runs/sr/<model>/...yaml` 即可复用现有数据、训练和评估流程。

新增 RefSR 模型时，在 `models/refsr/<ModelName>/` 放模型代码，在 `engines/refsr/` 增加只包含模型特有 step 的 engine，并添加对应 run 配置。公共 checkpoint、EMA、TensorBoard、scheduler 和 early stopping 逻辑应继续放在 `engines/base_trainer.py`。

模型目录不能直接依赖另一个具体模型目录的私有实现；跨模型复用的训练和运行时能力放在 `engines/`、`runtime/`、`losses/`、`metrics/`，WKV CUDA 代码放 `kernels/wkv/`。

## 验证

```bash
python3 -m compileall -q data engines models runtime scripts evaluation metrics losses tests
python tests/smoke_native_geometry.py --only x4
```

第一个命令不需要导入完整训练依赖即可检查语法；第二个命令需要 CUDA、编译器和 WKV 扩展。
