#!/bin/bash

cd /mnt/sda/home/zhangheyi/projects/RefRWKV

export CUDA_LAUNCH_BLOCKING=1

# 激活 conda 环境
source /mnt/sda/conda/miniforge3/etc/profile.d/conda.sh
conda activate rwkv7

# 运行训练脚本
python scripts/train_sr_prior.py \
  --config configs/runs/aid_x4_l1.yaml \
  --load_weights checkpoints/refrwkv_sr_aid_x4_l1/epoch=0173-val_loss=0.050643.ckpt
# 提交作业的命令（在终端中运行）:
# gpu-submit --name train_ref -- bash /mnt/sda/home/zhangheyi/projects/RefRWKV/scripts/submit_train.sh
# 查看 GPU 分区所有作业的命令:
# squeue -p gpu
# Submitted batch job 690
# Log file: /mnt/sda/home/zhangheyi/logs/slurm/train_ref-690.out
# Status: gpu-jobs
# Watch log: tail -f /mnt/sda/home/zhangheyi/logs/slurm/train_ref-690.out
# Cancel: scancel 690
# tensorboard --logdir logs