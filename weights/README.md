# Weights

`weights/pretrained/` stores reusable downloaded or published model weights.
`weights/exports/` stores deliberately exported inference artifacts. Training
checkpoints belong under `experiments/train/`, where they remain tied to the
exact config, optimizer state and dataset contract.

目录职责：

- `pretrained/`：外部下载或跨实验复用的预训练权重；
- `exports/`：明确导出的部署或分享权重。

训练产生的 `last.ckpt`、最佳 checkpoint 和 TensorBoard 日志统一放在
`experiments/train/.../`；测试图片和 `metrics.json` 放在
`experiments/test/.../`。
