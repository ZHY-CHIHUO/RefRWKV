# 测试快捷入口

此目录只放测试快捷脚本，与 `train/` 的训练入口分开。测试模型所需的网络、数据集和倍率配置全部从 checkpoint 内嵌的 `trainer_config` 恢复；测试策略来自独立的 `configs/test/test.yaml`。

## `run_test.sh`

对指定 checkpoint 执行 `test_easy`、`test_hard` 或其他配置中的 split，并写入 `experiments/test/`：

```bash
bash scripts/shortcuts/test/run_test.sh \
  experiments/train/refsr/refsrwkv/hrms_scd/x4/hrms_scd_trefsr_x4/checkpoints/last.ckpt

bash scripts/shortcuts/test/run_test.sh <checkpoint> \
  --split test_easy --metrics psnr ssim --no-save-images
```

第一个参数必须是 checkpoint，也可以通过 `REFRWKV_CHECKPOINT` 提供。测试 YAML 可以通过 `REFRWKV_TEST_CONFIG` 替换；其他参数原样传给 `scripts/test/run.py`。

Bicubic 是无参数基线，不需要 checkpoint；快捷脚本也支持直接提供 Bicubic 的
完整配置：

```bash
bash scripts/shortcuts/test/run_test.sh \
  --training-config configs/runs/sr/bicubic/hrms_scd_x4.yaml
```
