# `scripts/test/`

这里是统一测试入口。`run.py` 使用独立 `configs/test/test.yaml`，checkpoint 中保存的 `trainer_config` 提供模型、数据集、网络结构和倍率。

```bash
python scripts/test/run.py \
  --config configs/test/test.yaml \
  --checkpoint <experiments/train/.../last.ckpt>
```

`sr.py` 和 `refsr.py` 是兼容旧路径的转发入口，实际逻辑统一在 `run.py`。测试结果按 split 写到 `experiments/test/.../<split>/metrics.json`。Bicubic 使用参数免费基线路径运行，不加载 checkpoint 权重。

Bicubic 没有训练 checkpoint 时，显式提供它自己的完整训练配置作为数据和倍率元信息：

```bash
python scripts/test/run.py \
  --config configs/test/test.yaml \
  --training-config configs/runs/sr/bicubic/hrms_scd_x4.yaml
```

训练模型仍必须使用 `--checkpoint`；测试不会为了补齐模型或数据字段而回读
`configs/runs/`。
