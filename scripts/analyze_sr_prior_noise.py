#!/usr/bin/env python3
"""
分析 SR prior 图像质量相当于 HR 在潜空间加了多少噪声（扩散时间步 t）。

用法:
    python scripts/analyze_sr_prior_noise_latent.py --config configs/sd2_control_config.yaml

输出:
    打印基于潜空间 MSE 的等效 t，以及对应的统计。
"""

import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import yaml

# 添加项目路径
_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from diffusers import DDPMScheduler, AutoencoderKL
import lpips
from torch.utils.data import DataLoader

# 导入数据加载器和 SR 模型
from RefRWKV.models.RefSRWKV import RefSRWKV
from RefRWKV.RefSR_data.RefPNGDataset import RefPNGDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--device", type=str, default="cuda", help="设备")
    parser.add_argument(
        "--num_samples", type=int, default=None, help="限制样本数，用于快速测试"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 读取配置
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = args.device

    # 1. 加载 SD2 的 VAE（用于编码）
    sd_model_path = config["model"]["sd_model_path"]
    vae = AutoencoderKL.from_pretrained(
        sd_model_path, subfolder="vae", local_files_only=True
    )
    vae.to(device).eval()
    vae.requires_grad_(False)
    vae_scale_factor = vae.config.scaling_factor
    print(f"✅ Loaded VAE from {sd_model_path}")

    def encode_latent(img):
        """编码图像到潜空间，并乘以 scaling_factor"""
        with torch.no_grad():
            z = vae.encode(img).latent_dist.sample() * vae_scale_factor
        return z

    # 2. 加载 SR prior 模型
    sr_cfg = config["model"]["sr"]
    sr_model = RefSRWKV(
        inp_channels=sr_cfg["inp_channels"],
        out_channels=sr_cfg["out_channels"],
        dim=sr_cfg["dim"],
        num_blocks=sr_cfg["num_blocks"],
        num_refinement_blocks=sr_cfg["num_refinement_blocks"],
        scale=sr_cfg["scale"],
        drop_path_rate=sr_cfg["drop_path_rate"],
        hidden_rate=sr_cfg["hidden_rate"],
    )
    ckpt_path = sr_cfg["ckpt_path"]
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(f"SR prior checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("model_sr."):
            new_k = k[len("model_sr.") :]
            new_state_dict[new_k] = v
        else:
            new_state_dict[k] = v
    sr_model.load_state_dict(new_state_dict)
    sr_model.to(device).eval()
    print(f"✅ Loaded SR prior from {ckpt_path}")

    # 3. 加载验证数据集
    data_cfg = config["data"]
    dataset = RefPNGDataset(
        data_dir=data_cfg["root"],
        mode="val",
        patch_size=None,  # 用整图（避免裁剪引入额外误差）
        scale=data_cfg["scale"],
        augment=False,
        augment_ref=False,
    )
    if args.num_samples is not None:
        dataset.filenames = dataset.filenames[: args.num_samples]
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    print(f"📊 验证集样本数: {len(dataset)}")

    # 4. 加载扩散调度器
    scheduler = DDPMScheduler(
        num_train_timesteps=config["model"]["num_train_timesteps"],
        beta_start=config["model"]["beta_start"],
        beta_end=config["model"]["beta_end"],
        beta_schedule=config["model"]["beta_schedule"],
    )

    # 5. 准备 t 值列表（步长可调）
    t_values = torch.arange(0, 1000, 10, device=device, dtype=torch.long)

    # 6. 统计
    latent_mse_sr_list = []  # SR prior 与 HR 的 latent MSE
    equivalent_t_latent_list = []  # 基于 latent MSE 的等效 t
    lpips_sr_list = []  # 像素空间 LPIPS 辅助参考

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing")):
        lr = batch["lr"].to(device)
        ref = batch["ref"].to(device)
        hr = batch["hr"].to(device)

        with torch.no_grad():
            # 生成 SR prior
            sr = sr_model(lr, ref)  # 输出范围 [-1,1]

            # ---- 像素空间 LPIPS（仅供参考） ----
            lpips_fn = lpips.LPIPS(net="vgg").to(device)
            lpips_sr = lpips_fn(sr, hr).item()
            lpips_sr_list.append(lpips_sr)

            # ---- 潜空间分析 ----
            # 编码
            hr_latent = encode_latent(hr)
            sr_latent = encode_latent(sr)

            # 计算 SR prior latent 与 HR latent 的 MSE
            mse_sr = F.mse_loss(hr_latent, sr_latent).item()
            latent_mse_sr_list.append(mse_sr)

            # 寻找等效 t：在 HR latent 上加噪，使加噪后的 latent 与原始 HR latent 的 MSE 最接近 mse_sr
            best_t = 0
            best_diff = float("inf")
            for t_val in t_values:
                noise = torch.randn_like(hr_latent)
                t_tensor = torch.full((1,), t_val, device=device, dtype=torch.long)
                noisy_hr_latent = scheduler.add_noise(hr_latent, noise, t_tensor)
                mse_noisy = F.mse_loss(noisy_hr_latent, hr_latent).item()
                diff = abs(mse_noisy - mse_sr)
                if diff < best_diff:
                    best_diff = diff
                    best_t = t_val.item()
            equivalent_t_latent_list.append(best_t)

    # 7. 输出统计
    latent_mse_arr = np.array(latent_mse_sr_list)
    t_latent_arr = np.array(equivalent_t_latent_list)
    lpips_arr = np.array(lpips_sr_list)

    print("\n" + "=" * 60)
    print("📊 分析结果（基于潜空间 MSE）")
    print(f"  样本数: {len(latent_mse_arr)}")
    print(f"  SR prior latent MSE (vs HR latent):")
    print(f"    平均: {latent_mse_arr.mean():.6f} ± {latent_mse_arr.std():.6f}")
    print(f"    最小: {latent_mse_arr.min():.6f}, 最大: {latent_mse_arr.max():.6f}")
    print(f"  等效噪声时间步 t (基于潜空间 MSE):")
    print(f"    平均: {t_latent_arr.mean():.1f} ± {t_latent_arr.std():.1f}")
    print(f"    最小: {t_latent_arr.min():.0f}, 最大: {t_latent_arr.max():.0f}")
    print("\n📎 像素空间 LPIPS (仅供参考):")
    print(f"    平均: {lpips_arr.mean():.4f} ± {lpips_arr.std():.4f}")
    print("=" * 60)

    # 可选：画分布图
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        ax1, ax2, ax3, ax4 = axes.flatten()

        ax1.hist(latent_mse_arr, bins=20, alpha=0.7)
        ax1.set_title("SR prior latent MSE distribution")
        ax1.set_xlabel("MSE")
        ax1.set_ylabel("Count")

        ax2.hist(t_latent_arr, bins=20, alpha=0.7)
        ax2.set_title("Equivalent t (latent MSE) distribution")
        ax2.set_xlabel("t")
        ax2.set_ylabel("Count")

        ax3.hist(lpips_arr, bins=20, alpha=0.7)
        ax3.set_title("SR prior LPIPS (pixel) distribution")
        ax3.set_xlabel("LPIPS")
        ax3.set_ylabel("Count")

        # 散点图：latent MSE vs t
        ax4.scatter(latent_mse_arr, t_latent_arr, alpha=0.5)
        ax4.set_title("Latent MSE vs Equivalent t")
        ax4.set_xlabel("Latent MSE")
        ax4.set_ylabel("Equivalent t")

        plt.tight_layout()
        plt.savefig("sr_prior_noise_analysis_latent.png", dpi=150)
        print("📈 分布图已保存至 sr_prior_noise_analysis_latent.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
