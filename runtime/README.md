# `runtime/`

这里是与具体模型无关的运行时工具。

| 文件 | 内容 |
|---|---|
| `config.py` | YAML 加载、分层合并、override、完整配置物化和测试配置合并。 |
| `checkpoint.py` | checkpoint 读取、state dict/EMA 匹配、通道适配和 `trainer_config` 提取。 |
| `experiments.py` | 统一生成 `experiments/train` 和 `experiments/test` 路径。 |
| `common.py` | 路径解析、随机种子、JSON 安全转换等公共工具。 |
| `callbacks.py` | 训练回调和日志辅助。 |
| `tiling.py` | 大图 tile 推理和重叠拼接。 |

测试的关键约定：`load_test_config()` 只读取独立测试 YAML 的 `test` 段，然后覆盖 checkpoint 内嵌的训练配置中的同名测试策略；不会回到 `configs/runs/` 找训练配置。无参数的 Bicubic 不需要 checkpoint，由测试入口显式读取 `--training-config`。
