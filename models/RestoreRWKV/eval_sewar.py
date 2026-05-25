import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split, ConcatDataset
from models.RestoreRWKV.Restore_RWKV import Restore_RWKV_Ref
from data.dataset_ref import RefPairedEnviMemmapDataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 128

# 直接指向包含 .dat 文件的数据集目录
data_dir = "../../data/raw/data0"  # 根据实际情况修改

ref_dataset = RefPairedEnviMemmapDataset(
    data_dir=data_dir,
    patch_size=128,
    scale=8,
    mode="train",
    stride_ratio=0.5,
    use_memmap=True,
)

# 可选子集（如果数据量太大）
ref_dataset = Subset(ref_dataset, range(int(0.1 * len(ref_dataset))))

n_total = len(ref_dataset)
n_train = int(0.7 * n_total)
n_val = int(0.2 * n_total)
n_test = n_total - n_train - n_val

train_dataset, val_dataset, test_dataset = random_split(
    ref_dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(42)
)

test_loader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
)

print("数据集加载完成")

model = Restore_RWKV_Ref(inp_channels=8, out_channels=4, scale=8).to(device)
criterion = nn.L1Loss()

# -----------------------------
# 测试评估
# -----------------------------
from evaluation.eval_sewar import evaluate_CHW, average_metrics, print_metrics

model.load_state_dict(torch.load("best_restore_rwkv_ref.pth"))
model.eval()
results_hr2 = []  # 保存 output vs hr2 的指标
results_hr1 = []  # 保存 output vs hr1 的指标
with torch.no_grad():
    for lr1, hr1, lr2, hr2 in test_loader:
        lr1 = lr1.to(device)
        hr1 = hr1.to(device)
        lr2 = lr2.to(device)
        hr2 = hr2.to(device)
        output = model(lr1, hr1, lr2)

        for i in range(lr1.size(0)):
            pred_img = output[i].cpu().numpy()  # (C, H, W)
            gt_img = hr2[i].cpu().numpy()  # ground truth
            ref_img = hr1[i].cpu().numpy()  # reference

            # 计算与目标 HR2 的指标
            metrics_hr2 = evaluate_CHW(pred_img, gt_img, print_result=False)
            results_hr2.append(metrics_hr2)

            # 计算与参考 HR1 的指标
            metrics_hr1 = evaluate_CHW(pred_img, ref_img, print_result=False)
            results_hr1.append(metrics_hr1)

# 分别统计并打印
avg_metrics_hr2 = average_metrics(results_hr2)
avg_metrics_hr1 = average_metrics(results_hr1)

print_metrics(avg_metrics_hr2, title="Test Set Average Metrics (output vs HR2)")
print_metrics(avg_metrics_hr1, title="Test Set Average Metrics (output vs HR1)")
