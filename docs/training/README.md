# 训练文档

此目录存放训练命令、数据准备、消融实验、显存设置和运行记录。按任务或实验主题建立文档，命令中的配置路径应与 `configs/runs/` 对应。

常用训练快捷脚本位于 [`scripts/shortcuts/`](../../scripts/shortcuts/)，文件名格式为 `模型_数据集_sr/ref_倍率.sh`。HRMS-SCD 的三组命令可集中查看：

```bash
bash scripts/shortcuts/show_commands.sh
```

本地只打印某一条命令：

```bash
bash scripts/shortcuts/refsrwkv_hrms_scd_ref_x4.sh --print
```

远程集群提交时，将对应快捷脚本作为作业入口，例如：

```bash
gpu-submit --name hrms_scd_ref_x4 -- \
  bash /mnt/sda/home/zhangheyi/projects/RefRWKV/scripts/shortcuts/refsrwkv_hrms_scd_ref_x4.sh
```
