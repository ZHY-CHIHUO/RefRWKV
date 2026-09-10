# `experiments/`

这里保存训练和测试实验产出。仓库会提交可复核的元数据和分析结果，
但不提交预测图片、checkpoint 和训练日志等大文件。

## 提交范围

会提交：`config.yaml`、`config.json`、`test.yaml`、`metrics.json`、
`*.csv`、`*.json`、比较报告和协议文件。这些文件记录实际使用的配置、
逐图指标、配对统计和参考敏感性实验参数，便于复核跑过哪些实验。

不会提交：`images/`、`checkpoints/`、`logs/` 目录，以及 PNG/JPEG、
checkpoint 和日志文件。它们仍保留在本机的实验目录中。

## 训练

```text
experiments/train/<task>/<model>/<dataset>/x<scale>/<run>/
├── checkpoints/   # last.ckpt、最佳 checkpoint
├── logs/           # TensorBoard 和训练日志
├── config.yaml     # 最终可读训练配置
└── config.json     # JSON 快照
```

checkpoint 内的 `trainer_config` 是测试时恢复模型和数据配置的唯一训练配置来源。

## 测试

```text
experiments/test/<task>/<model>/<dataset>/x<scale>/<run>/<split>/
├── images/         # 按 sample_id 命名的预测 PNG
└── metrics.json    # 平均指标、逐图指标、sample_id 和测试元数据
```

测试根目录还会保存实际使用的 `test_easy.yaml`、`test_hard.yaml` 快照。这两个文件
就是后续测试的可编辑策略，下一次运行对应 split 时会优先读取它们；
`configs/test/test.yaml` 只作为首次生成模板。比较报告通常放在
`experiments/test/comparison/`。
