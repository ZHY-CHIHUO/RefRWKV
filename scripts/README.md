# `scripts/`

这里放训练、测试、数据准备、配置展开和结果分析的命令行入口。

| 路径 | 内容 |
|---|---|
| `train/` | SR、direct RefSR、RefSRWKV 和 RefDiffRWKV 训练入口。 |
| `test/` | 独立 YAML 驱动的统一测试入口。 |
| `shortcuts/train/` | 按模型/数据集/任务/倍率命名的训练 shell 快捷脚本。 |
| `shortcuts/test/` | 测试 shell 快捷脚本，当前为 `run_test.sh`。 |
| `prepare/` | 遥感图像和 Wuhan TIFF 数据准备。 |
| `evaluate.py` | 兼容旧路径的统一测试转发入口。 |
| `compare_refsr.py` | 按 `sample_id` 生成逐图 ΔPSNR、改善比例、配对 t 检验和直方图。 |
| `reference_quality_scan.py` | 推理时扫描参考图质量并输出增益曲线。 |
| `reference_sensitivity_aid.py` | HRMS checkpoint 在 AID 上的零样本参考敏感性扫描（光谱、光度和位移扰动）。 |
| `render_config.py` | 将分层训练 YAML 展开为可编辑完整配置。 |
| `submit.sh`、`submit_train.sh` | 通用提交说明和训练快捷脚本转发入口。 |

## 训练和测试

训练快捷脚本位于 `shortcuts/train/`，例如：

```bash
bash scripts/shortcuts/train/swinir_hrms_scd_sr_x4.sh
bash scripts/shortcuts/train/refsrwkv_hrms_scd_ref_x4.sh --print
bash scripts/shortcuts/train/refsrwkv_pancollection_wv3.sh --print
bash scripts/shortcuts/train/show_commands.sh
```

测试快捷脚本位于 `shortcuts/test/`，训练模型以 checkpoint 为输入：

```bash
bash scripts/shortcuts/test/run_test.sh \
  experiments/train/refsr/refsrwkv/hrms_scd/x4/hrms_scd_trefsr_x4/checkpoints/last.ckpt
```

测试策略由 `configs/test/test.yaml` 控制，训练配置从 checkpoint 内嵌的 `trainer_config` 读取；不要在测试时重新指定 `configs/runs/` 训练 YAML。

Bicubic 是例外的无参数基线，可用 `--training-config configs/runs/sr/bicubic/hrms_scd_x4.yaml`
提供数据和倍率元信息。

AID 参考敏感性实验使用固定 500 张 `test` 子集；先用 HRMS RefSRWKV-SR checkpoint
得到无参考基线，再由扫描脚本默认加载 HRMS TRefSR checkpoint：

```bash
conda run --no-capture-output -n rwkv7 python scripts/test/run.py \
  --config configs/test/aid_reference_sensitivity.yaml \
  --checkpoint experiments/train/refsr/refsrwkv/hrms_scd/x4/hrms_scd_sr_x4/checkpoints/last.ckpt \
  --output experiments/test/refsr/refsrwkv/aid/x4/hrms_scd_sr_x4_zero_shot \
  --split test --no-save-images

conda run --no-capture-output -n rwkv7 python scripts/reference_sensitivity_aid.py
```

扫描结果写入 `experiments/test/comparison/aid_reference_sensitivity/`；协议和逐图结果
会保留在该目录，便于复核 `p=0` 控制与 `p=0.2...1.0` 五档结果。
