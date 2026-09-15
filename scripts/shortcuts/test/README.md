# 测试快捷入口

此目录只放测试快捷脚本，与 `train/` 的训练入口分开。测试模型所需的网络、数据集和倍率配置全部从 checkpoint 内嵌的 `trainer_config` 恢复；测试策略来自独立的 `configs/test/test.yaml`。

## `run_test.sh`

对指定 checkpoint 执行 `test_easy`、`test_hard` 或其他配置中的 split，并写入 `experiments/test/`：

```bash
bash scripts/shortcuts/test/run_test.sh \
  experiments/train/<task>/<model>/<dataset>/x<scale>/<run>/checkpoints/last.ckpt

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

官方 FusionMamba WV3 权重使用独立快捷脚本测试。默认权重路径是
`/tmp/FusionMamba2/weights/420.pth`，也可以显式传入权重文件：

```bash
bash scripts/shortcuts/test/fusion_mamba_pancollection_wv3.sh \
  /path/to/420.pth --batch-size 16 --no-save-images
```

该适配器使用全图 `256x256` HR 输入，数据和模型都走 `[0, 1]` 范围。测试数据根目录仍由
`configs/datasets/refsr/pancollection_wv3.yaml` 中的 PanCollection H5 配置决定。

## `bi_wkv_cuda.sh`

本机检查原版双向 Bi-WKV CUDA 能否编译、前反向是否有限：

```bash
bash scripts/shortcuts/test/bi_wkv_cuda.sh
bash scripts/shortcuts/test/bi_wkv_cuda.sh --unittest
```

## `compare_wv3_pan.sh`

WV3 上对比 FusionMamba 与 RDM-PAN，指标与论文表格一致：

- Reduced：`PSNR Q2n SAM ERGAS`
- Full-res：`Dλ Ds QNR`（Ds 使用 toolbox 1.0 的 MATLAB `imresize`）

```bash
bash scripts/shortcuts/test/compare_wv3_pan.sh
bash scripts/shortcuts/test/compare_wv3_pan.sh fusion_mamba
bash scripts/shortcuts/test/compare_wv3_pan.sh rdm_pan
```

默认权重是各自 `last.ckpt`。可用环境变量覆盖：

- `FUSION_MAMBA_CKPT`
- `RDM_PAN_CKPT`
- `REFRWKV_OUTPUT`（默认 `experiments/test/refsr/compare_wv3_pan`）
- `REFRWKV_PYTHON`
