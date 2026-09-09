# `engines/`

这里实现训练生命周期和各任务的 train/validation step。模型构造放在 `models/`，配置和 checkpoint 管理放在 `runtime/`，engine 只负责把它们接到训练框架。

| 路径 | 内容 |
|---|---|
| `base_trainer.py` | 公共 optimizer、EMA、验证、scheduler、checkpoint 和日志生命周期。 |
| `sr/trainer.py` | 单图 SR 的 batch 解包、前向和损失。 |
| `refsr/trainer.py` | direct RefSR 模型的 `forward(lr, ref)` 训练步骤。 |
| `refsr/refsrwkv_trainer.py` | RefSRWKV 的参考图、融合和损失逻辑。 |
| `refsr/refdiff_trainer.py` | RefDiffRWKV 的扩散、生成器/判别器和梯度流程。 |

训练入口在 `scripts/train/`，不要在 engine 中重复实现数据扫描或测试输出逻辑。
