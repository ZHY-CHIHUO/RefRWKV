# `models/`

这里保存 SR、RefSR 和 RefDiffRWKV 的网络实现及 registry/adapter。训练 engine 只依赖统一的模型接口，不直接复制某个模型的训练循环。

## 目录

| 路径 | 内容 |
|---|---|
| `sr/` | 单图 SR registry、SwinIR，以及 Bicubic、EDSR、RCAN、HAT、MambaIRv2 compatibility baseline。 |
| `refsr/` | direct RefSR registry、RefSRWKV、TTSR、MASA-SR、DATSR 和 RefDiffRWKV。 |

SR 模型通常提供 `forward(lr)`；direct RefSR 模型提供 `forward(lr, ref)`。模型输入统一遵守项目的 `[-1, 1]` tensor 契约，Bicubic 是无参数的 `reference_only` 评测基线。

## 模型家族

- `models/sr/swinir/`：项目现有 SwinIR 网络和 adapter。
- `models/sr/baselines.py`、`baseline_adapters.py`：轻量单图对比模型和 registry 适配。
- `models/refsr/refsrwkv/`：参考图超分 RWKV，支持 `paired` 真实 Ref 和 `lr_up` 自参考两种数据模式。
- `models/refsr/baselines.py`、`baseline_adapters.py`：TTSR、MASA-SR、DATSR 的统一 direct RefSR 适配。
- `models/refsr/RefDiffRWKV/`：扩散生成器、SR prior、参考适配器、判别器和采样器。

HRMS-SCD x4 的共同比较协议在 `configs/common/benchmark.yaml`；模型配置在 `configs/models/`，一次实验组合在 `configs/runs/`。官方实现与本仓库的 `native_compatibility` 结果不能混用。
