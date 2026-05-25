import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split, ConcatDataset
from Restore_RWKV import Restore_RWKV_Ref

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from RWKV.data.dataset_ref import RefPairedEnviMemmapDataset

torch.cuda.set_sync_debug_mode(0)   # 1 = 同步检查，2 = 设备端断言

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
batch_size = 16
num_epochs = 50

data_dir = '../../data/raw/data0'

ref_dataset = RefPairedEnviMemmapDataset(
    data_dir=data_dir,
    patch_size=256,
    scale=8,
    mode='train',
    stride_ratio=0.5,
    use_memmap=True
)

# 固定子集与划分（种子固定，多次运行一致）
ref_dataset = Subset(ref_dataset, range(int(1.0 * len(ref_dataset))))
n_total = len(ref_dataset)
n_train = int(0.7 * n_total)
n_val   = int(0.2 * n_total)
n_test  = n_total - n_train - n_val

train_dataset, val_dataset, test_dataset = random_split(
    ref_dataset, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          num_workers=4, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)

print("数据集加载完成")

model = Restore_RWKV_Ref(inp_channels=8, out_channels=4, scale=8).to(device)
criterion = nn.L1Loss()
lr_max = 1e-4
warmup_steps = 100
optimizer = torch.optim.Adam(model.parameters(), lr=lr_max)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# -----------------------------
# 断点续训设置
# -----------------------------
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
latest_ckpt_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')

start_epoch = 1
global_step = 0
best_val_loss = float('inf')
early_stop_counter = 0
early_stop_patience = 5

if os.path.exists(latest_ckpt_path):
    print(f"发现存档 {latest_ckpt_path}，加载中...")
    checkpoint = torch.load(latest_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1   # 从下一个 epoch 继续
    global_step = checkpoint['global_step']
    best_val_loss = checkpoint['best_val_loss']
    early_stop_counter = checkpoint['early_stop_counter']
    print(f"已从 Epoch {checkpoint['epoch']} 恢复，global_step = {global_step}, best_val_loss = {best_val_loss:.4f}")
else:
    print("未找到存档，从头开始训练。")

# -----------------------------
# 训练/验证循环
# -----------------------------
for epoch in range(start_epoch, num_epochs + 1):
    # ---------- 训练 ----------
    model.train()
    train_loss = 0.0
    for step, (lr1, hr1, lr2, hr2) in enumerate(train_loader):
        lr1 = lr1.to(device)
        hr1 = hr1.to(device)
        lr2 = lr2.to(device)
        hr2 = hr2.to(device)

        optimizer.zero_grad()
        output = model(lr1, hr1, lr2)
        loss = criterion(output, hr2)
        loss.backward()
        optimizer.step()

        # Warm-up：仅在尚未超过 warmup 步数时调整学习率
        global_step += 1
        if global_step <= warmup_steps:
            warmup_lr = lr_max * global_step / warmup_steps
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr

        train_loss += loss.item() * lr1.size(0)
        if step % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}/{num_epochs} Step {step}/{len(train_loader)} "
                  f"Loss: {loss.item():.4f} LR: {current_lr:.2e}")

    train_loss /= len(train_loader.dataset)

    # ---------- 验证 ----------
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for lr1, hr1, lr2, hr2 in val_loader:
            lr1 = lr1.to(device)
            hr1 = hr1.to(device)
            lr2 = lr2.to(device)
            hr2 = hr2.to(device)
            output = model(lr1, hr1, lr2)
            loss = criterion(output, hr2)
            val_loss += loss.item() * lr1.size(0)
        val_loss /= len(val_loader.dataset)

    print(f"Epoch [{epoch}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    scheduler.step(val_loss)

    # ---------- 早停判断 ----------
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), 'best_restore_rwkv_ref.pth')
        print(f"  Saved new best model with val loss {best_val_loss:.4f}")
    else:
        early_stop_counter += 1
        print(f"  No improvement. Early stop counter: {early_stop_counter}/{early_stop_patience}")
        if early_stop_counter >= early_stop_patience:
            print(f"Early stopping triggered after {early_stop_patience} epochs without improvement.")
            # 触发早停前仍保存一次最新存档，方便之后可能继续
            torch.save({
                'epoch': epoch,
                'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'early_stop_counter': early_stop_counter,
            }, latest_ckpt_path)
            break

    # ---------- 保存最新存档 ----------
    torch.save({
        'epoch': epoch,
        'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'early_stop_counter': early_stop_counter,
    }, latest_ckpt_path)
    print(f"  Checkpoint saved at {latest_ckpt_path}")

# -----------------------------
# 测试评估
# -----------------------------
from RWKV.evaluation.eval_sewar import *

# 测试前加载最佳模型（或存档中的最后模型）
if os.path.exists('best_restore_rwkv_ref.pth'):
    model.load_state_dict(torch.load('best_restore_rwkv_ref.pth'))
else:
    # 若没有最佳模型（例如从未改善），则使用最后一次存档的模型
    checkpoint = torch.load(latest_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
results_list = []

with torch.no_grad():
    for lr1, hr1, lr2, hr2 in test_loader:
        lr1 = lr1.to(device)
        hr1 = hr1.to(device)
        lr2 = lr2.to(device)
        hr2 = hr2.to(device)
        output = model(lr1, hr1, lr2)

        for i in range(lr1.size(0)):
            pred_img = output[i].cpu().numpy()
            gt_img   = hr2[i].cpu().numpy()
            metrics = evaluate_CHW(pred_img, gt_img, print_result=False)
            results_list.append(metrics)

avg_metrics = average_metrics(results_list)
print_metrics(avg_metrics, title="Test Set Average Metrics")