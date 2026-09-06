# 训练快捷入口

文件名格式为 `模型_数据集_任务_倍率.sh`。脚本从自身位置解析项目根目录，因此本地和远程使用同一份文件。

三个按模型命名的脚本负责实际训练；`show_commands.sh` 只负责集中打印三条命令，不会启动训练。

## HRMS-SCD 三组训练

| 脚本 | 训练内容 | 数据参考模式 |
|---|---|---|
| `swinir_hrms_scd_sr_x4.sh` | SwinIR SISR | 不使用参考图 |
| `refsrwkv_hrms_scd_sr_x4.sh` | RefSRWKV SISR | `lr_up`，由 LR 生成自参考 |
| `refsrwkv_hrms_scd_ref_x4.sh` | RefSRWKV TRefSR | `paired`，读取跨时相 `Ref/` |

远程服务器上，在项目根目录执行脚本即可启动训练：

```bash
bash scripts/shortcuts/swinir_hrms_scd_sr_x4.sh
bash scripts/shortcuts/refsrwkv_hrms_scd_sr_x4.sh
bash scripts/shortcuts/refsrwkv_hrms_scd_ref_x4.sh
```

也可以继续使用统一的远程提交入口；传入快捷脚本名即可切换实验：

```bash
bash scripts/submit_train.sh swinir_hrms_scd_sr_x4.sh
bash scripts/submit_train.sh refsrwkv_hrms_scd_sr_x4.sh
bash scripts/submit_train.sh refsrwkv_hrms_scd_ref_x4.sh
```

例如通过集群提交 TRefSR：

```bash
gpu-submit --name hrms_scd_ref_x4 -- \
  bash /mnt/sda/home/zhangheyi/projects/RefRWKV/scripts/submit_train.sh \
  refsrwkv_hrms_scd_ref_x4.sh
```

查看统一入口支持的脚本：

```bash
bash scripts/submit_train.sh --list
```

脚本检测到未激活环境时，会尝试激活 `/mnt/sda/conda/miniforge3` 下的 `rwkv7`。如果远程环境路径不同，可以覆盖：

```bash
REFRWKV_CONDA_SH=/path/to/conda.sh \
REFRWKV_CONDA_ENV=rwkv7 \
bash scripts/shortcuts/refsrwkv_hrms_scd_ref_x4.sh
```

本地只查看命令，不会构造模型或启动训练：

```bash
bash scripts/shortcuts/swinir_hrms_scd_sr_x4.sh --print
bash scripts/shortcuts/refsrwkv_hrms_scd_sr_x4.sh --print
bash scripts/shortcuts/refsrwkv_hrms_scd_ref_x4.sh --print
```

一次查看三组命令：

```bash
bash scripts/shortcuts/show_commands.sh
```

查看带消融覆盖的命令：

```bash
bash scripts/shortcuts/show_commands.sh \
  --overrides model.fusion_match.enabled=false model.global_latent_blocks=0
```

脚本支持继续训练和命令行覆盖：

```bash
bash scripts/shortcuts/refsrwkv_hrms_scd_ref_x4.sh \
  --resume experiments/train/refsr/RefSRWKV/hrms_scd/x4/hrms_scd_trefsr_x4/checkpoints/last.ckpt

bash scripts/shortcuts/refsrwkv_hrms_scd_sr_x4.sh \
  --overrides train.max_steps=1000 data.batch_size=2
```

本地若要实际运行，可先激活包含项目依赖的环境，或设置 `REFRWKV_PYTHON` 指向 Python 可执行文件。
