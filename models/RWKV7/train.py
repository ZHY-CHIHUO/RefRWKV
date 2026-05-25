import os
import glob
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Subset, random_split
from model import RWKV

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from RWKV.data.dataset_ref import RefPairedEnviMemmapDataset


torch.cuda.set_sync_debug_mode(0)  # 关闭同步调试以加速训练
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('medium')   # 加速 bf16 训练


def get_latest_checkpoint(checkpoint_dir):
    """
    自动寻找可用的断点续训文件。
    优先使用 Lightning 自动保存的 last.ckpt；
    若不存在，则扫描目录获取最新的 .ckpt 文件。
    若目录中没有任何 .ckpt，返回 None。
    """
    # 优先使用固定的断点文件
    last_ckpt = os.path.join(checkpoint_dir, "last.ckpt")
    if os.path.exists(last_ckpt):
        print(f"✅ 自动检测到断点续训文件: {last_ckpt}")
        return last_ckpt

    # 否则找目录里最新的 .ckpt
    all_ckpts = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if all_ckpts:
        latest_ckpt = max(all_ckpts, key=os.path.getmtime)
        print(f"✅ 自动检测到最新 checkpoint: {latest_ckpt}")
        return latest_ckpt

    print("🔹 未找到任何 checkpoint，将从头开始训练。")
    return None


def main():
    # ======================== 1. 超参数配置 ========================
    class Args:
        pass
    args = Args()
    # 模型结构
    args.n_embd = 1024
    args.n_layer = 8
    args.head_size = 64          # 需与 CUDA 内核编译时的 HEAD_SIZE 一致
    args.my_testing = "x070"
    args.grad_cp = 0             # 梯度检查点（0 为关闭）
    # 优化器
    args.lr_init = 1e-3
    args.betas = (0.9, 0.99)
    args.adam_eps = 1e-6
    args.weight_decay = 0.1
    args.epochs = 200             # 总训练 epoch 数
    # 图像相关参数
    args.hr_size = 256
    args.lr_size = 32
    args.patch_size_hr = 16
    args.patch_size_lr = 8
    # 残差训练参数
    args.ss_prob = 0.2

    # 数据加载
    batch_size = 32
    num_workers = 4
    data_dir = '../../data/raw/data0'

    # ======================== 2. 数据集准备 ========================
    full_dataset = RefPairedEnviMemmapDataset(
        data_dir=data_dir,
        patch_size=256,
        scale=8,
        mode='train',
        stride_ratio=0.5,
        use_memmap=True
    )

    # 固定子集（若希望使用全部数据，可将比例设为1.0）
    full_dataset = Subset(full_dataset, range(int(0.7 * len(full_dataset))))
    n_total = len(full_dataset)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    n_test = n_total - n_train - n_val

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)  # 固定划分种子
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    print(f"数据集划分：train={n_train}, val={n_val}, test={n_test}")

    # ======================== 3. 模型初始化 ========================
    model = RWKV(args)
    # 若需要加载预训练权重（非断点续训），可在此执行：
    # model.load_state_dict(torch.load("pretrain.pth", map_location="cpu"), strict=False)

    # ======================== 4. 回调与日志 ========================
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 模型保存：保留验证损失最优的3个，并总是保存最新一份用于断点续训
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="rwkv-sr-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,           # 关键！生成 last.ckpt，用于自动续训
    )

    # 早停策略
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        mode="min",
        verbose=True,
    )

    # 学习率监控
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # TensorBoard 日志
    logger = TensorBoardLogger("./logs", name="rwkv_sr")

    # ======================== 5. Trainer 配置 ========================
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=1,
        precision="bf16-true",          # 匹配 RWKV7 CUDA 内核的 bf16 要求
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        # 多卡训练时可启用 Deepspeed 策略:
        # strategy="deepspeed_stage_2",
    )

    # ======================== 6. 自动续训并开始训练 ========================
    resume_ckpt = get_latest_checkpoint(checkpoint_dir)
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_ckpt)

    # ======================== 7. 测试阶段：自动使用最优权重 ========================
    best_ckpt = checkpoint_callback.best_model_path
    trainer.test(model, test_loader, ckpt_path=best_ckpt)

    # trainer.test(model, test_loader, ckpt_path="checkpoints/rwkv-sr-epoch=04-val_loss=0.0270.ckpt")

    print(args)


if __name__ == "__main__":
    main()