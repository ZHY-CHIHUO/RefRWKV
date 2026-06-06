import os
import glob
import torch
from argparse import Namespace
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    LearningRateMonitor,
    EarlyStopping,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader, Subset, random_split
from model import RWKVSR  # 修改1：导入正确的 LightningModule

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from RWKV.data.dataset_ref import RefPairedEnviMemmapDataset

torch.cuda.set_sync_debug_mode(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_float32_matmul_precision("medium")


def get_latest_checkpoint(checkpoint_dir):
    last_ckpt = os.path.join(checkpoint_dir, "last.ckpt")
    if os.path.exists(last_ckpt):
        print(f"✅ 自动检测到断点续训文件: {last_ckpt}")
        return last_ckpt
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
    args.n_embd = 64
    args.dim_att = args.n_embd
    args.head_size = 64    
    args.n_layer = 4
    args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32)
    args.my_testing = "x070"
    args.grad_cp = 0

    # channelrwkv_args
    args.channel_rwkv_args = Namespace(
        n_embd=64,
        dim_att=64,
        head_size=64,
        n_layer=2,
        grad_cp=0,
        my_testing='x070'
    )

    # 空间（瓶颈） RWKV 配置
    args.spatial_rwkv_args = Namespace(
        n_embd=64,
        dim_att=64,
        head_size=64,
        n_layer=4,
        grad_cp=0,
        my_testing='x070'
    )

    # 优化器
    args.lr_init = 1e-4
    args.betas = (0.9, 0.99)
    args.adam_eps = 1e-6
    args.weight_decay = 0.1
    args.epochs = 200

    # 图像相关参数
    args.hr_size = 256
    args.lr_size = 32
    args.patch_size_hr = 16
    args.patch_size_lr = 8
    args.ss_prob = 0.0

    # 数据加载
    batch_size = 8
    num_workers = 4

    # ---------- 新增：数据目录列表与划分比例 ----------
    data_dirs = ["../../data/raw/data0"]   # 可添加更多目录，如 "../../data/raw/data1"
    ratios = (0.7, 0.15, 0.15)            # 训练:验证:测试 空间行比例
    global_norm_path = "./norm_stats_global.json"   # 全局归一化文件（需预先合并生成）

    # ======================== 2. 数据集准备 ========================
    from torch.utils.data import ConcatDataset

    # 如果全局归一化文件不存在，会退回到各目录自己的 norm_stats.json，但混合训练时不推荐
    norm_file = global_norm_path if Path(global_norm_path).exists() else None

    # 训练集
    train_subsets = []
    for d in data_dirs:
        ds = RefPairedEnviMemmapDataset(
            data_dir=d,
            patch_size=256,
            scale=8,
            mode="train",
            stride_ratio=0.5,
            use_memmap=True,
            augment=True,
            ratios=ratios,
            norm_stats_file=norm_file
        )
        train_subsets.append(ds)
    train_dataset = ConcatDataset(train_subsets) if len(train_subsets) > 1 else train_subsets[0]

    # 验证集
    val_subsets = []
    for d in data_dirs:
        ds = RefPairedEnviMemmapDataset(
            data_dir=d,
            patch_size=256,
            scale=8,
            mode="val",
            use_memmap=True,
            augment=False,
            ratios=ratios,
            norm_stats_file=norm_file
        )
        val_subsets.append(ds)
    val_dataset = ConcatDataset(val_subsets) if len(val_subsets) > 1 else val_subsets[0]

    # 测试集
    test_subsets = []
    for d in data_dirs:
        ds = RefPairedEnviMemmapDataset(
            data_dir=d,
            patch_size=256,
            scale=8,
            mode="test",
            use_memmap=True,
            augment=False,
            ratios=ratios,
            norm_stats_file=norm_file
        )
        test_subsets.append(ds)
    test_dataset = ConcatDataset(test_subsets) if len(test_subsets) > 1 else test_subsets[0]

    print(f"数据集划分结果：训练={len(train_dataset)}, 验证={len(val_dataset)}, 测试={len(test_dataset)}")

    # ---------- 新增：collate_fn（含随机互换）----------
    import random

    def ref_swap_collate_fn(batch, swap_prob=0.5):
        # batch 是 list of tuples: (lr1, hr1, lr2, hr2)
        lr1_batch, hr1_batch, lr2_batch, hr2_batch = zip(*batch)
        lr1 = torch.stack(lr1_batch, 0)
        hr1 = torch.stack(hr1_batch, 0)
        lr2 = torch.stack(lr2_batch, 0)
        hr2 = torch.stack(hr2_batch, 0)

        if random.random() < swap_prob:
            # 交换参考和目标
            return lr2, hr2, lr1, hr1
        else:
            return lr1, hr1, lr2, hr2

    def no_swap_collate_fn(batch):
        lr1_batch, hr1_batch, lr2_batch, hr2_batch = zip(*batch)
        lr1 = torch.stack(lr1_batch, 0)
        hr1 = torch.stack(hr1_batch, 0)
        lr2 = torch.stack(lr2_batch, 0)
        hr2 = torch.stack(hr2_batch, 0)
        return lr1, hr1, lr2, hr2

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=lambda b: ref_swap_collate_fn(b, swap_prob=0.5)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=no_swap_collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=no_swap_collate_fn
    )

    # ======================== 3. 模型初始化 ========================
    model = RWKVSR(args)

    # ======================== 4. 回调与日志 ========================
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="rwkv-sr-{epoch:02d}-{val_loss:.6f}",
        monitor="val_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=10,
        min_delta=1e-4,
        mode="min",
        verbose=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    logger = TensorBoardLogger("./logs", name="rwkv_sr")

    # ======================== 5. Trainer 配置 ========================
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu",
        devices=1,
        precision="bf16-true",
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        accumulate_grad_batches=4,
    )

    # ======================== 6. 自动续训并开始训练 ========================
    resume_ckpt = get_latest_checkpoint(checkpoint_dir)
    trainer.fit(model, train_loader, val_loader, ckpt_path=resume_ckpt)

    # ======================== 7. 测试 ========================
    best_ckpt = checkpoint_callback.best_model_path
    trainer.test(model, test_loader, ckpt_path=best_ckpt)

    print(args)

if __name__ == "__main__":
    main()
