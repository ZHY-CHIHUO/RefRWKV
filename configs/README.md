# `configs/`

这里保存训练和测试所需的 YAML 配置。配置按职责分层，训练 run 文件通过 `base` 引用公共默认值；测试配置独立于训练配置。

## 目录

| 目录 | 内容 |
|---|---|
| `common/` | 设备、优化器、损失、数据契约和基线协议等公共默认值。 |
| `datasets/` | 数据集根目录、split、尺寸、倍率和 LR 来源元数据。 |
| `models/` | 网络结构默认参数。 |
| `runs/` | 可直接交给 `scripts/train/*.py` 的分层训练入口。 |
| `test/` | 独立测试策略；当前入口是 `test/test.yaml`。 |

## 训练配置

训练配置依次合并 `common/`、`datasets/`、`models/` 和 `runs/` 中的内容。需要得到可编辑的完整 YAML 时运行：

```bash
python scripts/render_config.py \
  --config configs/runs/refsrwkv/hrms_scd_trefsr_x4.yaml
```

完整快照会写到对应的 `experiments/train/.../config.yaml`。训练 checkpoint 也会保存最终的 `trainer_config`。

### `base` 覆盖顺序

`base` 列表按从上到下的顺序合并：后面的文件覆盖前面的文件；当前 YAML
文件自身最后覆盖所有 `base`。mapping 会递归合并，同名标量、列表或整个子节点
则由后者替换。例如：

```yaml
base:
  - ../../../common/sr.yaml
  - ../../../models/sr/swinir_m.yaml
  - ../../../common/benchmark.yaml
```

等价于：

```text
当前 run YAML > benchmark.yaml > swinir_m.yaml > sr.yaml
```

因此 benchmark 放在最后可以锁定公共对比协议；如果 run 文件再次写同一个字段，
run 文件的值仍然优先。

## 测试配置

`configs/test/test.yaml` 只控制：

```yaml
test:
  splits: [test_easy, test_hard]
  metrics: [psnr, ssim]
  save_images: true
  device: null
  batch_size: null
  steps: null
  output: null
```

测试时使用 `scripts/test/run.py --config configs/test/test.yaml --checkpoint <ckpt>`。模型、数据集、倍率和网络结构从 checkpoint 内嵌的最终训练 YAML 读取，不会重新从 `configs/runs/` 加载训练配置。

测试开始后，实际策略会快照到测试输出根目录：
`experiments/test/.../<run>/test_easy.yaml` 和 `test_hard.yaml`。这些文件是可编辑的，
下一次测试会优先读取对应 split 的快照；`configs/test/test.yaml` 只负责首次生成默认值。

## 常用约定

- `data.*` 描述数据路径、split、采样和键名。
- `model.*` 描述网络结构和参考图分支。
- `train.*` 描述优化器、验证、checkpoint 和设备。
- `loss.*` 描述训练目标。
- `output.*` 描述实验输出根目录；通常由运行时自动生成。
- HRMS-SCD x4 的统一比较协议在 `common/benchmark.yaml`。
