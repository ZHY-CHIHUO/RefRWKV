# `scripts/shortcuts/`

快捷脚本按训练和测试分开：

| 目录 | 内容 |
|---|---|
| `train/` | 模型、数据集、任务和倍率固定的训练 `.sh` 入口。 |
| `test/` | 使用 checkpoint 和独立测试 YAML 的测试 `.sh` 入口。 |
| `_common.sh` | 训练快捷脚本共享的项目根目录、环境和参数转发逻辑。 |

训练脚本的具体列表和命令见 [`train/README.md`](train/README.md)。测试脚本见 [`test/README.md`](test/README.md)。Bicubic 没有训练脚本，作为无参数测试基线运行。
