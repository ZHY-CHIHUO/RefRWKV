# 训练文档

此目录存放训练命令、数据准备、消融实验、显存设置和运行记录。按任务或实验主题建立文档，命令中的配置路径应与 `configs/runs/` 对应。

常用训练快捷脚本位于 [`scripts/shortcuts/`](../../scripts/shortcuts/)，文件名格式为 `模型_数据集_sr/ref_倍率.sh`。HRMS-SCD 的三组命令可集中查看：

```bash
bash scripts/shortcuts/show_commands.sh
```

完整的 HRMS-SCD x4 基线协议、全部快捷入口和评估规范见
[模型对比表](../models/baselines.md)。`scripts/submit_train.sh --list` 会列出
新增的 EDSR、RCAN、HAT、MambaIRv2、TTSR、MASA-SR 和 DATSR 快捷脚本；Bicubic
是无参数评估基线，使用 `scripts/test/sr.py --split test_easy|test_hard`，不训练。

本地只打印某一条命令：

```bash
bash scripts/shortcuts/refsrwkv_hrms_scd_ref_x4.sh --print
```

远程集群提交时，将对应快捷脚本作为作业入口，例如：

```bash
gpu-submit --name hrms_scd_ref_x4 -- \
  bash /mnt/sda/home/zhangheyi/projects/RefRWKV/scripts/shortcuts/refsrwkv_hrms_scd_ref_x4.sh
```
