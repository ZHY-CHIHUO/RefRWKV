#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
RefRWKV 测试脚本
Better Start (RefSRWKV) → SDEdit 扩散去噪 → EnRWKV 增强 → IQA 评估
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image

# 项目路径（根据实际结构调整）
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from RefRWKV.models.RefSRWKV import RefSRWKV
from RefRWKV.models.RefDiffRWKV import RefDiffRWKV
from RefRWKV.models.EnRWKV import EnRWKV
from RefRWKV.models.GlobalSemanticModule import GlobalSemanticModule
from RefRWKV.RefRWKV_PL import RefRWKV_PL
from RefRWKV.RefSR_data.RefPNGDataset import RefPNGDataset
from RefRWKV.evaluation.eval_pyiqa import IQAEngine


# -------------------- 扩散工具函数 --------------------
def get_diffusion_schedule(num_timesteps, device):
    """与训练一致的余弦 schedule"""
    s = 0.008
    T = num_timesteps
    t = torch.arange(0, T, device=device, dtype=torch.float32)
    alpha_bar = torch.cos(((t / T + s) / (1 + s)) * np.pi / 2) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    betas = torch.cat([betas, 1 - alpha_bar[-1:]])
    betas = torch.clamp(betas, max=0.999)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return alphas, alpha_bars, betas


def add_noise(x0, noise, alpha_bar_t):
    sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
    sqrt_one_minus = torch.sqrt(1 - alpha_bar_t)
    return sqrt_alpha_bar * x0 + sqrt_one_minus * noise


@torch.no_grad()
def sdedit_sample(model_diff, x_start, lr, ref, num_timesteps,
                  start_step, sampling_steps, device):
    B = x_start.shape[0]
    alphas, alpha_bars, betas = get_diffusion_schedule(num_timesteps, device)

    # 1. 加噪到 start_step
    noise = torch.randn_like(x_start)
    alpha_bar_start = alpha_bars[start_step].view(-1, 1, 1, 1)
    x_t = add_noise(x_start, noise, alpha_bar_start)

    # 2. 构建去噪时间步
    if start_step >= sampling_steps:
        # linspace 产生从 start_step 到 0 的采样步数+1 个均匀点
        raw_times = torch.linspace(start_step, 0, sampling_steps + 1,
                                device=device).long()
        # 去重 + 翻转得到降序排列（避免 sorted=True 的兼容性问题）
        times = torch.unique(raw_times).flip(0)
    else:
        # start_step < sampling_steps: 逐步下降
        times = torch.arange(start_step, -1, -1, device=device)


    times = times[1:]
    if times[-1] != 0:
        times = torch.cat([times, torch.tensor([0], device=device)])

    # 3. 逐步去噪
    for t in tqdm(times, desc='SDEdit denoising'):
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        noise_pred = model_diff(x_t, t_batch, lr, ref)

        alpha = alphas[t].view(-1, 1, 1, 1)
        alpha_bar = alpha_bars[t].view(-1, 1, 1, 1)
        beta = betas[t].view(-1, 1, 1, 1)

        x_t = (1 / torch.sqrt(alpha)) * (
            x_t - (1 - alpha) / torch.sqrt(1 - alpha_bar) * noise_pred
        )
        if t > 0:
            noise = torch.randn_like(x_t)
            x_t = x_t + torch.sqrt(beta) * noise
    return x_t


# -------------------- 主函数 --------------------
def main():
    parser = argparse.ArgumentParser(description='RefRWKV Test')
    parser.add_argument('--config', type=str, default='configs/test_config.yaml',
                        help='测试配置文件')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='覆盖配置文件中的 checkpoint 路径')
    parser.add_argument('--output', type=str, default=None,
                        help='覆盖输出目录')
    parser.add_argument('--ntest', type=int, default=None,
                        help='覆盖测试样本数')
    parser.add_argument('--save_images', action='store_true', default=None,
                        help='覆盖是否保存图像')
    parser.add_argument('--no_save', action='store_true',
                        help='强制不保存图像')
    parser.add_argument('--device', type=str, default=None,
                        help='覆盖设备')
    parser.add_argument('--sampling_steps', type=int, default=None)
    parser.add_argument('--diff_start_step', type=int, default=None)
    parser.add_argument('--use_sr', action='store_true', default=None)
    parser.add_argument('--no_sr', action='store_true')
    parser.add_argument('--use_diff', action='store_true', default=None)
    parser.add_argument('--no_diff', action='store_true')
    parser.add_argument('--use_enhance', action='store_true', default=None)
    parser.add_argument('--no_enhance', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    # ---------- 加载配置文件 ----------
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    test_cfg = cfg.get('test', {})
    model_cfg = cfg.get('model', {})
    sr_cfg = cfg.get('sr', {})
    enhance_cfg = cfg.get('enhance', {})
    data_cfg = cfg['data']

    # 命令行参数覆盖配置文件
    checkpoint = args.checkpoint or test_cfg.get('checkpoint', 'checkpoints/last.ckpt')
    output_dir = args.output or test_cfg.get('output_dir', 'results/test')
    device_str = args.device or test_cfg.get('device', 'cuda')
    save_images = args.save_images if args.save_images is not None else test_cfg.get('save_images', True)
    if args.no_save:
        save_images = False

    ntest = args.ntest if args.ntest is not None else test_cfg.get('ntest', None)
    seed = args.seed if args.seed is not None else test_cfg.get('seed', 42)

    use_sr = test_cfg.get('use_sr', True)
    if args.use_sr:
        use_sr = True
    if args.no_sr:
        use_sr = False

    use_diff = test_cfg.get('use_diff', True)
    if args.use_diff:
        use_diff = True
    if args.no_diff:
        use_diff = False

    use_enhance = test_cfg.get('use_enhance', True)
    if args.use_enhance:
        use_enhance = True
    if args.no_enhance:
        use_enhance = False

    sampling_steps = args.sampling_steps if args.sampling_steps is not None else test_cfg.get('sampling_steps', 50)
    diff_start_step = args.diff_start_step if args.diff_start_step is not None else test_cfg.get('diff_start_step', 200)
    num_timesteps = test_cfg.get('num_timesteps', 1000)

    # ---------- 随机种子 ----------
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ---------- 构建模型 ----------
    global_semantic = GlobalSemanticModule(base_dim=model_cfg.get('embed_dim', 64))

    model_diff = RefDiffRWKV(
        img_size=model_cfg.get('img_size', 256),
        patch_size=model_cfg.get('patch_size', 4),
        embed_dim=model_cfg.get('embed_dim', 64),
        channels=model_cfg.get('channels', 3),
        enc_blocks=model_cfg.get('enc_blocks', [4, 6, 6]),
        dec_blocks=model_cfg.get('dec_blocks', [6, 6, 4]),
        latent_blocks=model_cfg.get('latent_blocks', 8),
        drop_path_rate=model_cfg.get('drop_path_rate', 0.1),
        hidden_rate=model_cfg.get('hidden_rate', 4),
        learn_sigma=model_cfg.get('learn_sigma', False),
        upsample_mode=model_cfg.get('upsample_mode', 'cnn'),
        global_semantic=global_semantic,
    )
    model_sr = RefSRWKV(
        inp_channels=sr_cfg.get('inp_channels', 3),
        out_channels=sr_cfg.get('out_channels', 3),
        dim=sr_cfg.get('dim', 48),
        num_blocks=sr_cfg.get('num_blocks', [4, 6, 6, 8]),
        num_refinement_blocks=sr_cfg.get('num_refinement_blocks', 8),
        scale=sr_cfg.get('scale', 10),
    )
    model_enhance = EnRWKV(
        inp_channels=enhance_cfg.get('inp_channels', 3),
        out_channels=enhance_cfg.get('out_channels', 3),
        dim=enhance_cfg.get('dim', 48),
        num_blocks=enhance_cfg.get('num_blocks', [4, 6, 6, 8]),
        num_refinement_blocks=enhance_cfg.get('num_refinement_blocks', 4),
    )

    pl_model = RefRWKV_PL.load_from_checkpoint(
        checkpoint,
        model_sr=model_sr,
        model_diff=model_diff,
        model_enhance=model_enhance,
        strict=False,
    )
    device = torch.device(device_str)
    pl_model.to(device)
    pl_model.eval()

    # 如果配置未显式指定开关，且 checkpoint 中有训练标志，可回退到 checkpoint 配置（这里直接使用我们的变量，用户可控）
    # 若用户想依据训练开关自动决定，可自行添加逻辑；此处保持显式可控

    # ---------- 测试数据集 ----------
    test_ds = RefPNGDataset(
        mode='test',
        data_dir=data_cfg['root'],
        patch_size=data_cfg.get('patch_size'),   # 测试时设为 None 可处理全图
        scale=data_cfg.get('scale', 10),
        ref_aug_strengths=data_cfg.get('ref_aug_strengths', [0.12, 0.12, 0.12, 0.03]),
        ref_aug_probs=data_cfg.get('ref_aug_probs', [0.5, 0.5, 0.5, 0.5]),
        ref_gray_prob=data_cfg.get('ref_gray_prob', 0.2),
        max_samples=(data_cfg.get('max_samples_train'),
                     data_cfg.get('max_samples_val'),
                     data_cfg.get('max_samples_test')),
        sample_seed=seed,
        augment=False,
        augment_ref=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=data_cfg.get('batch_size', 1),
        shuffle=False,
        num_workers=data_cfg.get('num_workers', 4),
        pin_memory=True,
    )

    # 输出目录
    out_dir = Path(output_dir)
    img_dir = out_dir / 'images'
    gt_dir = out_dir / 'gt'
    sr_dir = out_dir / 'sr'
    if save_images:
        img_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        sr_dir.mkdir(parents=True, exist_ok=True) 

    img_size = model_cfg.get('img_size', 256)
    diff_start_step = min(diff_start_step, num_timesteps - 1)

    print(f"Configuration summary:")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Output dir: {output_dir}")
    print(f"  Device: {device_str}")
    print(f"  SR={use_sr}, Diffusion={use_diff}, Enhance={use_enhance}")
    print(f"  SDEdit: start_step={diff_start_step}, sampling_steps={sampling_steps}")
    print(f"  N test: {ntest}, Save images: {save_images}")

    # ---------- 推理 ----------
    idx = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Testing'):
            lr, hr, ref = batch
            lr = lr.to(device)
            hr = hr.to(device)
            ref = ref.to(device)

            if ntest is not None and idx >= ntest:
                break

            # Step 1: RefSRWKV 超分
            if use_sr:
                I_start = pl_model.model_sr(lr, ref, label=None)
            else:
                I_start = F.interpolate(lr, size=(img_size, img_size),
                                        mode='bicubic', align_corners=False)
            
            if save_images:                
                I_start_save = (I_start + 1.0) / 2.0                
                I_start_save = torch.clamp(I_start_save, 0.0, 1.0)                
                sr_np = (I_start_save[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)                
                Image.fromarray(sr_np).save(sr_dir / f"{idx:05d}.png")

            # Step 2: 扩散去噪
            if use_diff and sampling_steps > 0:
                I_out = sdedit_sample(
                    pl_model.model_diff,
                    I_start, lr, ref,
                    num_timesteps, diff_start_step, sampling_steps, device,
                )
            else:
                I_out = I_start

            # Step 3: 增强
            if use_enhance:
                I_out = pl_model.model_enhance(I_out, label=None)

            # 回到 [0,1]
            I_out = (I_out + 1.0) / 2.0
            I_out = torch.clamp(I_out, 0.0, 1.0)

            # 保存
            if save_images:
                out_np = (I_out[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(out_np).save(img_dir / f"{idx:05d}.png")
                gt_np = (hr[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(gt_np).save(gt_dir / f"{idx:05d}.png")

            idx += 1

    print(f"Done. Processed {idx} images.")

    # ---------- IQA 评估 ----------
    if save_images and idx > 0 and test_cfg.get('iqa_enabled', True):
        print("Starting IQA evaluation...")
        engine = IQAEngine(device=device_str)
        engine.evaluate_folder(
            pred_dir=str(img_dir),
            gt_dir=str(gt_dir) if gt_dir.exists() else None,
            ntest=None,
            out_path=str(out_dir),
        )
    elif not save_images:
        print("Images not saved, skipping IQA. Use --save_images or enable in config.")
    else:
        print("No images processed, skipping IQA.")


if __name__ == '__main__':
    main()