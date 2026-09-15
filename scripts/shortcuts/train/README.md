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
| `ttsr_hrms_scd_ref_x4.sh` | TTSR 真实参考图 RefSR。 |
| `masa_sr_hrms_scd_ref_x4.sh` | MASA-SR 真实参考图 RefSR。 |
| `datsr_hrms_scd_ref_x4.sh` | DATSR 真实参考图 RefSR。 |
| `fusion_mamba_pancollection_wv3.sh` | FusionMamba WV3 全色融合训练，batch 32，官方 64x64 crop / L1 / Adam / StepLR / 500 epoch。 |
| `fusion_mamba_pancollection_wv3_official.sh` | 同上，保留官方命名入口。 |
| `rdm_refsr_pancollection_wv3.sh` | RDM-PAN WV3 全色融合训练，batch 32，训练协议与 FusionMamba 对齐。 |
| `rdm_pan_pancollection_wv3_v2.sh` | RDM-PAN WV3 v2：4× 全色融合，batch 64，800 epoch，几何增强 + 光谱置零。 |
| `rdm_pan_pancollection_qb_v2.sh` | RDM-PAN QuickBird v2：4 波段 MS + PAN，配方与 WV3 v2 相同。 |
| `stf_mamba_wuhan.sh` | 官方 STFMamba 的 4 波段 Wuhan 四元组训练，`(C0,F0,C1)->F1`，`[0,1]`。 |
| `rdm_stf_wuhan.sh` | RDM-STF Wuhan 四元组训练，输入契约与 STFMamba 相同。 |

## 用法

```bash
# FusionMamba WV3
bash scripts/shortcuts/train/fusion_mamba_pancollection_wv3.sh

# RDM-PAN WV3
bash scripts/shortcuts/train/rdm_refsr_pancollection_wv3.sh
bash scripts/shortcuts/train/rdm_pan_pancollection_wv3_v2.sh

# RDM-PAN QuickBird
bash scripts/shortcuts/train/rdm_pan_pancollection_qb_v2.sh

# STFMamba / RDM-STF Wuhan
bash scripts/shortcuts/train/stf_mamba_wuhan.sh
bash scripts/shortcuts/train/rdm_stf_wuhan.sh

# 只打印最终命令
bash scripts/shortcuts/train/rdm_refsr_pancollection_wv3.sh --print

# 通过统一提交入口选择脚本
bash scripts/submit_train.sh rdm_refsr_pancollection_wv3.sh
bash scripts/submit_train.sh --list
```

脚本支持 `--resume`、`--load-weights` 和 `--overrides` 等训练入口参数。若对应实验目录已经有 `experiments/train/.../config.yaml`，训练入口会优先使用该完整快照；因此直接编辑该文件即可改变后续训练配置。
