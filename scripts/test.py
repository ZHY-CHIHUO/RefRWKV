#!/usr/bin/env python
# test_noise_mse_x0.py
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
import numpy as np
from PIL import Image

# 添加项目根目录
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from RWKV.models.RefRWKV.RefDiffRWKV import RefDiffRWKV, RefDiffRWKV_PL
from RWKV.RefSR_data.RefSR_dataset import RefPNGDataset


def add_noise(hr, t, num_timesteps=1000, s=0.008):
    """余弦噪声调度"""
    alpha_bar = (
        torch.cos(((t.float() / num_timesteps + s) / (1 + s)) * torch.pi / 2) ** 2
    )
    alpha_bar = alpha_bar.view(-1, 1, 1, 1)
    noise = torch.randn_like(hr)
    x_t = torch.sqrt(alpha_bar) * hr + torch.sqrt(1 - alpha_bar) * noise
    return x_t, noise, alpha_bar


def test_fixed_t_mse(model, test_loader, device, num_timesteps=1000):
    """测试2: 固定 t 的 noise MSE 曲线"""
    t_list = list(range(0, 1000, 10))  # 0, 10, 20, ..., 990
    # 如果你希望包含 999 作为终点，可以补充：
    t_list.append(999)
    mse_dict = {t: 0.0 for t in t_list}
    count_dict = {t: 0 for t in t_list}

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing fixed-t noise MSE"):
            lr, hr, ref = batch
            lr = lr.to(device, dtype=torch.float32)
            hr = hr.to(device, dtype=torch.float32)
            ref = ref.to(device, dtype=torch.float32)
            B = hr.shape[0]

            for t_val in t_list:
                t = torch.full((B,), t_val, device=device, dtype=torch.long)
                x_t, noise, _ = add_noise(hr, t, num_timesteps=num_timesteps)
                pred_noise = model(x_t, t, lr, ref)
                loss = F.mse_loss(pred_noise, noise, reduction="mean")
                mse_dict[t_val] += loss.item() * B
                count_dict[t_val] += B

    mse_dict = {t: mse_dict[t] / count_dict[t] for t in t_list}
    print("\nFixed-t Noise MSE:")
    for t in t_list:
        print(f"t={t}: MSE={mse_dict[t]:.6f}")

    return mse_dict


def test_x0_recovery(model, test_loader, device, num_timesteps=1000):
    """测试3: x0 预测能力（PSNR/MSE）"""
    psnr_list = []
    mse_list = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing x0 recovery"):
            lr, hr, ref = batch
            lr = lr.to(device, dtype=torch.float32)
            hr = hr.to(device, dtype=torch.float32)
            ref = ref.to(device, dtype=torch.float32)
            B = hr.shape[0]

            # 随机 t
            t = torch.randint(0, num_timesteps, (B,), device=device)
            x_t, noise, alpha_bar = add_noise(hr, t, num_timesteps=num_timesteps)
            pred_noise = model(x_t, t, lr, ref)

            # 预测 x0
            pred_x0 = (x_t - torch.sqrt(1 - alpha_bar) * pred_noise) / torch.sqrt(
                alpha_bar
            )
            mse = F.mse_loss(pred_x0, hr, reduction="mean").item()
            mse_list.append(mse)

            # PSNR
            mse_pixel = ((pred_x0 - hr) ** 2).mean().item()
            psnr = 10 * np.log10(1.0 / mse_pixel)
            psnr_list.append(psnr)

    avg_mse = np.mean(mse_list)
    avg_psnr = np.mean(psnr_list)
    print(f"\nX0 Recovery - Avg MSE: {avg_mse:.6f}, Avg PSNR: {avg_psnr:.2f} dB")
    return avg_mse, avg_psnr


def main():
    # 配置
    model_ckpt = "checkpoints/last-v1.ckpt"
    data_root = "/home/zhy/PROJECT/RWKV/RefSR_data/ALL_2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 1

    # 加载模型
    base_model = RefDiffRWKV(
        img_size=480,
        patch_size=4,
        embed_dim=64,
        enc_blocks=[4, 6, 6],
        dec_blocks=[6, 6, 4],
        latent_blocks=8,
        drop_path_rate=0.0,
        upsample_mode="cnn",
        channels=3,
    )
    pl_model = RefDiffRWKV_PL.load_from_checkpoint(
        model_ckpt, model=base_model, strict=False
    )
    model = pl_model.model.to(device).float()
    model.eval()

    # 测试数据集
    test_dataset = RefPNGDataset(
        data_dir=data_root,
        mode="test",
        patch_size=160,
        augment=False,
        max_samples=(None, None, 100),
        sample_seed=42,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 测试2: 固定 t 的 noise MSE
    test_fixed_t_mse(model, test_loader, device)

    # 测试3: x0 recovery
    test_x0_recovery(model, test_loader, device)


if __name__ == "__main__":
    main()
