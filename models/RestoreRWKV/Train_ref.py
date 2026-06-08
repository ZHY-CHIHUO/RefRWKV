import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from Restore_RWKV import LitRestoreRWKV_Ref
from RWKV.RefSR_data.RefSR_dataset import RefPNGDataset  # 你的数据集类

# ----------------- 配置 -----------------
data_root = r"/home/zhy/PROJECT/RWKV/RefSR_data/ALL_2"
batch_size = 16
accumulate_grad_batches = 4  # 梯度累积步数，原 accumulation_steps
num_epochs = 200
num_workers = 2
scale = 10
patch_size = 160

max_samples = (10000, None, None)  # train 限制2000？实际写了10000


# ----------------- 数据集构建 -----------------
def get_dataloaders():
    train_dataset = RefPNGDataset(
        data_dir=data_root,
        mode="train",
        patch_size=patch_size,
        scale=scale,
        augment=True,
        max_samples=max_samples,
        sample_seed=42,
    )
    val_dataset = RefPNGDataset(
        data_dir=data_root,
        mode="val",
        patch_size=None,
        scale=scale,
        augment=False,
        max_samples=max_samples,
        sample_seed=42,
    )
    test_dataset = RefPNGDataset(
        data_dir=data_root,
        mode="test",
        patch_size=None,
        scale=scale,
        augment=False,
        max_samples=max_samples,
        sample_seed=42,
    )

    print(
        f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    def ref_collate_fn(batch):
        lr_list, hr_list, ref_list = zip(*batch)
        lr = torch.stack(lr_list, 0)
        hr = torch.stack(hr_list, 0)
        ref = torch.stack(ref_list, 0)
        # 返回 (lr1, hr1, lr2, hr2) 其中 lr2 也使用原 lr
        return lr, ref, lr, hr

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=ref_collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=ref_collate_fn,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=ref_collate_fn,
    )
    return train_loader, val_loader, test_loader


# ----------------- 模型 -----------------
model = LitRestoreRWKV_Ref(
    inp_channels=3, out_channels=3, scale=scale, learning_rate=1e-4, warmup_steps=100
)

train_loader, val_loader, test_loader = get_dataloaders()
model.set_dataloaders(train_loader, val_loader, test_loader)

# ----------------- 回调 -----------------
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="best-{epoch:02d}-{val_loss:.4f}",
    monitor="val_loss",
    mode="min",
    save_top_k=3,
    save_last=True,
)
early_stop_callback = EarlyStopping(monitor="val_loss", patience=10, mode="min")
lr_monitor = LearningRateMonitor(logging_interval="step")
logger = TensorBoardLogger("lightning_logs", name="ref_sr_experiment")

# ----------------- Trainer -----------------
trainer = pl.Trainer(
    max_epochs=num_epochs,
    accelerator="auto",
    devices=1,
    accumulate_grad_batches=accumulate_grad_batches,
    callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
    logger=logger,
    log_every_n_steps=10,
    gradient_clip_val=0.5,  # 可选梯度裁剪
)

# 开始训练
trainer.fit(model)

# ----------------- 调用 eval.py 进行测试 -----------------
import subprocess
import sys

best_ckpt_path = checkpoint_callback.best_model_path or "checkpoints/last.ckpt"

eval_cmd = [
    sys.executable,
    "eval.py",  # 确保 eval.py 在可执行路径
    "--weight",
    best_ckpt_path,
    "--data_root",
    data_root,
    "--device",
    "cuda:0",
    "--batch_size",
    str(batch_size),
    "--scale",
    str(scale),
    "--output_dir",
    "test_results",
]

print(f"Starting evaluation: {' '.join(eval_cmd)}")
subprocess.run(eval_cmd, check=True)
print("Evaluation finished. Results saved to test_results/")
