#!/bin/bash
# 进入包含 cuda 文件夹的正确目录
cd /mnt/sda/home/zhangheyi/RWKV/models/RestoreRWKV

# 显存碎片优化（可选）
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 直接使用 rwkv7 环境中的 Python 运行训练脚本
/mnt/sda/home/zhangheyi/.conda/envs/rwkv7/bin/python /mnt/sda/home/zhangheyi/RWKV/Train_ref.py

# 提交作业的命令（在终端中运行）:
# gpu-submit --name train_ref -- bash /mnt/sda/home/zhangheyi/RWKV/submit_train.sh
# 查看 GPU 分区所有作业的命令:
# squeue -p gpu
# 下载远程服务器文件到本地的命令:
# scp -r 4090:/mnt/sda/home/zhangheyi/projects/RWKV/eval.py .