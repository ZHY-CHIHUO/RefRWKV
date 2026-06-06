import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from RWKV.data.dataset_ref import RefPairedEnviMemmapDataset
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from Restore_RWKV import Restore_RWKV_Ref          # 请确保模型文件存在
from RWKV.RefSR_data.RefSR_dataset import RefPNGDataset              # 你新写的数据集类

# ----------------- 配置 -----------------
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_root = r"/home/zhy/PROJECT/RWKV/RefSR_data/ALL_2"
batch_size = 8
accumulation_steps = 4
num_epochs = 200
num_workers = 0
scale = 10
patch_size = 480

# 控制各集数量：train 取2000张，val/test 全取
max_samples = (2000, None, None)

# ----------------- 数据集构建 -----------------
train_dataset = RefPNGDataset(
    data_dir=data_root, mode="train",
    patch_size=patch_size, scale=scale, augment=True,
    max_samples=max_samples, sample_seed=42
)
val_dataset = RefPNGDataset(
    data_dir=data_root, mode="val",
    patch_size=None, scale=scale, augment=False,
    max_samples=max_samples, sample_seed=42
)
test_dataset = RefPNGDataset(
    data_dir=data_root, mode="test",
    patch_size=None, scale=scale, augment=False,
    max_samples=max_samples, sample_seed=42
)

print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

# ----- 重组为 (lr1, hr1, lr2, hr2) -----
def ref_collate_fn(batch):
    lr_list, hr_list, ref_list = zip(*batch)
    lr = torch.stack(lr_list, 0)          # [B,3,H_lr,W_lr]
    hr = torch.stack(hr_list, 0)          # [B,3,H_hr,W_hr]
    ref = torch.stack(ref_list, 0)        # [B,3,H_hr,W_hr]
    # 映射到模型输入输出：lr1=lr, hr1=ref, lr2=lr, hr2=hr
    return lr, ref, lr, hr

# DataLoader
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True,
    num_workers=num_workers, pin_memory=True,
    drop_last=True, collate_fn=ref_collate_fn
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=True,
    collate_fn=ref_collate_fn
)
test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=True,
    collate_fn=ref_collate_fn
)

# ----------------- 模型 -----------------
model = Restore_RWKV_Ref(inp_channels=3, out_channels=3, scale=scale).to(device)
criterion = nn.L1Loss()
lr_max = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=lr_max)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5
)

# ----------------- 断点续训 -----------------
checkpoint_dir = 'checkpoints'
os.makedirs(checkpoint_dir, exist_ok=True)
latest_ckpt_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')

start_epoch = 1
global_step = 0
best_val_loss = float('inf')
early_stop_counter = 0
early_stop_patience = 10

if os.path.exists(latest_ckpt_path):
    checkpoint = torch.load(latest_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    global_step = checkpoint['global_step']
    best_val_loss = checkpoint['best_val_loss']
    early_stop_counter = checkpoint['early_stop_counter']
    print(f"Resumed from epoch {checkpoint['epoch']}")

# ----------------- 训练循环 -----------------
for epoch in range(start_epoch, num_epochs + 1):
    model.train()
    train_loss = 0.0
    optimizer.zero_grad()                         # 每个累积周期开始时清零

    for step, (lr1, hr1, lr2, hr2) in enumerate(train_loader):
        lr1, hr1, lr2, hr2 = lr1.to(device), hr1.to(device), lr2.to(device), hr2.to(device)

        output = model(lr1, hr1, lr2)
        loss = criterion(output, hr2)

        # 梯度累积：将 loss 除以累积步数，反向传播时梯度自动叠加
        loss = loss / accumulation_steps
        loss.backward()

        # 每 accumulation_steps 步才更新一次
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

            # warmup 逻辑也放在这里（基于实际更新次数）
            global_step += 1
            if global_step <= 100:
                warmup_lr = lr_max * global_step / 100
                for pg in optimizer.param_groups:
                    pg['lr'] = warmup_lr

        # 记录 loss 时使用原始 loss（未除以 accumulation_steps）
        train_loss += loss.item() * accumulation_steps * lr1.size(0)

        if step % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}/{num_epochs} Step {step} Loss {loss.item() * accumulation_steps:.4f} "
                  f"LR: {current_lr:.2e}")

    # 处理可能不满 accumulation_steps 的尾部数据
    if (step + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()

    train_loss /= len(train_loader.dataset)

    # 验证
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for lr1, hr1, lr2, hr2 in val_loader:
            lr1, hr1, lr2, hr2 = lr1.to(device), hr1.to(device), lr2.to(device), hr2.to(device)
            output = model(lr1, hr1, lr2)
            val_loss += criterion(output, hr2).item() * lr1.size(0)
    val_loss /= len(val_loader.dataset)

    print(f"Epoch [{epoch}/{num_epochs}] Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")
    scheduler.step(val_loss)

    # 早停与保存
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        early_stop_counter = 0
        torch.save(model.state_dict(), 'best_restore_rwkv_ref.pth')
        print(f"  New best model (val loss {best_val_loss:.4f})")
    else:
        early_stop_counter += 1
        if early_stop_counter >= early_stop_patience:
            print("Early stopping triggered.")
            torch.save({
                'epoch': epoch, 'global_step': global_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'early_stop_counter': early_stop_counter
            }, latest_ckpt_path)
            break

    torch.save({
        'epoch': epoch, 'global_step': global_step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_loss': best_val_loss,
        'early_stop_counter': early_stop_counter
    }, latest_ckpt_path)

# ----------------- 测试 -----------------
from RWKV.evaluation.eval_sewar import *   # 保持原有评估

if os.path.exists('best_restore_rwkv_ref.pth'):
    model.load_state_dict(torch.load('best_restore_rwkv_ref.pth'))
else:
    checkpoint = torch.load(latest_ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
results_list = []
with torch.no_grad():
    for lr1, hr1, lr2, hr2 in test_loader:
        lr1, hr1, lr2, hr2 = lr1.to(device), hr1.to(device), lr2.to(device), hr2.to(device)
        output = model(lr1, hr1, lr2)
        for i in range(lr1.size(0)):
            pred = output[i].cpu().numpy()
            gt = hr2[i].cpu().numpy()
            results_list.append(evaluate_CHW(pred, gt, print_result=False))

avg = average_metrics(results_list)
print_metrics(avg, title="Test Set Average Metrics")