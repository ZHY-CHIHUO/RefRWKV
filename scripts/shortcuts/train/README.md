# 训练快捷脚本

这里的 `.sh` 文件都是训练入口，文件名按“模型_数据集_任务_倍率”命名。每个脚本开头都标注了用途；它们通过共享的 `../_common.sh` 定位项目根目录、准备 Python 环境并调用 `scripts/train/`。

## 可用脚本

| 脚本 | 用途 |
|---|---|
| `swinir_hrms_scd_sr_x4.sh` | HRMS-SCD x4 SwinIR 单图 SR。 |
| `edsr_hrms_scd_sr_x4.sh` | HRMS-SCD x4 EDSR 单图 SR。 |
| `rcan_hrms_scd_sr_x4.sh` | HRMS-SCD x4 RCAN 单图 SR。 |
| `hat_hrms_scd_sr_x4.sh` | HRMS-SCD x4 HAT 单图 SR。 |
| `mambairv2_hrms_scd_sr_x4.sh` | HRMS-SCD x4 MambaIRv2 单图 SR。 |
| `refsrwkv_hrms_scd_sr_x4.sh` | RefSRWKV-SR，`lr_up` 自参考。 |
| `refsrwkv_hrms_scd_ref_x4.sh` | RefSRWKV TRefSR，读取配对真实 `Ref/`。 |
| `ttsr_hrms_scd_ref_x4.sh` | TTSR 真实参考图 RefSR。 |
| `masa_sr_hrms_scd_ref_x4.sh` | MASA-SR 真实参考图 RefSR。 |
| `datsr_hrms_scd_ref_x4.sh` | DATSR 真实参考图 RefSR。 |
| `refsrwkv_wuhan.sh` | Wuhan 四通道 temporal-pair 训练，网络倍率 x1。 |

## 用法

```bash
# 直接训练
bash scripts/shortcuts/train/swinir_hrms_scd_sr_x4.sh

# 只打印最终命令
bash scripts/shortcuts/train/refsrwkv_hrms_scd_ref_x4.sh --print

# 打印所有训练命令
bash scripts/shortcuts/train/show_commands.sh

# 通过统一提交入口选择脚本
bash scripts/submit_train.sh rcan_hrms_scd_sr_x4.sh
bash scripts/submit_train.sh --list
```

脚本支持 `--resume`、`--load-weights` 和 `--overrides` 等训练入口参数。若对应实验目录已经有 `experiments/train/.../config.yaml`，训练入口会优先使用该完整快照；因此直接编辑该文件即可改变后续训练配置。
