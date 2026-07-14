#!/usr/bin/env python
"""
全面测试脚本：Better Start t_start 对比 + 轨迹分析

功能：
  1. 标准 benchmark：不同 t_start 下完整 DDIM 采样的最终 PSNR
  2. 轨迹对比：SR 路径(有adapter) vs HR 路径(无adapter) 的逐步去噪轨迹
     - 输出每步的 SR PSNR, HR PSNR, Δ(SR-HR), CROSS PSNR(SR vs HR), 时间步 t
     - 分析轨迹收敛性
  3. [可选] MSE Guidance 测试
  4. [可选] 保存中间图像

用法:
    # 纯 benchmark
    python scripts/benchmark_tstart.py \
        --config configs/sd2_ref_gan_config.yaml \
        --ckpt checkpoints/sd2_ref_gan/last.ckpt \
        --num_samples 30 --sample_steps 100 \
        --t_start_list 0 300 400 500 600 700

    # 轨迹对比
    python scripts/benchmark_tstart.py \
        --config configs/sd2_ref_gan_config.yaml \
        --ckpt checkpoints/sd2_ref_gan/last.ckpt \
        --num_samples 50 --sample_steps 100 \
        --t_start_list 30 50 100 200 300 400 500 600 700 800 900 999 --compare_trajectory

    # 轨迹对比 + 保存图像（每10步保存一张）
    python scripts/benchmark_tstart.py \
        --config configs/sd2_ref_gan_config.yaml \
        --ckpt checkpoints/sd2_ref_gan/last.ckpt \
        --num_samples 5 --sample_steps 100 \
        --t_start_list 300 600 \
        --compare_trajectory --save_images --save_interval 10

    # 测试 MSE Guidance
    python scripts/benchmark_tstart.py \
        --config configs/sd2_ref_gan_config.yaml \
        --ckpt checkpoints/sd2_ref_gan/last.ckpt \
        --num_samples 30 --sample_steps 100 \
        --t_start_list 300 \
        --guidance_scale_list 0.0 0.05 0.1 0.2
"""

import sys
import argparse
import yaml
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

from RefRWKV.RefSR_data.RefSR_dataset import RefLMDBDataset
from RefRWKV.models.RefDiffRWKV.sd2_ref_generator import SD2RefGenerator
from RefRWKV.models.RefSRWKV import RefSRWKV


def parse_args():
    parser = argparse.ArgumentParser(
        description="全面测试 Better Start / 轨迹对比",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--ckpt", type=str, required=True, help="RefGAN checkpoint 路径")
    parser.add_argument("--num_samples", type=int, default=30, help="测试样本数")
    parser.add_argument("--sample_steps", type=int, default=100, help="DDIM 步数")
    parser.add_argument(
        "--t_start_list", type=str, nargs="+",
        default=["0", "300", "400", "500", "600", "700"],
        help="要测试的 t_start 列表。'0' 或 'none' 表示纯噪声起点。",
    )
    parser.add_argument(
        "--compare_trajectory", action="store_true",
        help="逐步对比 SR vs HR 路径的去噪轨迹",
    )
    parser.add_argument(
        "--save_images", action="store_true",
        help="轨迹模式下保存中间图像（需配合 --compare_trajectory）",
    )
    parser.add_argument(
        "--save_interval", type=int, default=10,
        help="保存图像的步数间隔（默认每10步保存一张）",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="输出目录（默认自动生成时间戳目录）",
    )
    parser.add_argument(
        "--guidance_scale_list", type=float, nargs="+", default=None,
        help="测试不同 guidance_scale（MSE Guidance）对采样质量的影响。"
             "与 --t_start_list 组合使用。例: 0.0 0.05 0.1",
    )
    parser.add_argument(
        "--t_stop", type=int, default=0,
        help="MSE Guidance 仅在 t > t_stop 时启用（默认0）",
    )
    return parser.parse_args()


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_refgan(cfg, ckpt_path, device):
    mc = cfg.get("model", {})
    sr_cfg = mc.get("sr", {})
    sr_model = RefSRWKV(
        inp_channels=sr_cfg.get("inp_channels", 3),
        out_channels=sr_cfg.get("out_channels", 3),
        dim=sr_cfg.get("dim", 48),
        num_blocks=tuple(sr_cfg.get("num_blocks", [4, 6, 6, 8])),
        num_refinement_blocks=sr_cfg.get("num_refinement_blocks", 4),
        scale=sr_cfg.get("scale", 10),
    ).to(device).eval()
    for p in sr_model.parameters():
        p.requires_grad = False

    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("state_dict", ckpt)
    key_weight = state_dict.get(
        "generator.adapter.core.merge_encoder.lr_block1.spatial.key.weight",
        state_dict.get("adapter.core.merge_encoder.lr_block1.spatial.key.weight", None),
    )
    ckpt_embed_dim = key_weight.shape[0] if key_weight is not None else mc.get("rwkv_cfg", {}).get("embed_dim", 192)

    rwkv_cfg = dict(mc.get("rwkv_cfg", {"patch_size": 4, "embed_dim": 192}))
    rwkv_cfg["embed_dim"] = ckpt_embed_dim

    generator = SD2RefGenerator(
        lr_key=mc.get("lr_key", "lr"),
        ref_key=mc.get("ref_key", "ref"),
        hr_key=mc.get("hr_key", "hr"),
        sd_model_path=mc["sd_model_path"],
        use_lora=mc.get("use_lora", True),
        lora_rank=mc.get("lora_rank", 64),
        sd_locked=mc.get("sd_locked", True),
        sr_model=sr_model,
        use_sr_latent_cond=mc.get("use_sr_latent_cond", False),
        rwkv_cfg=rwkv_cfg,
        strategy=mc.get("strategy", "rwkv"),
        use_semantic=mc.get("use_semantic", True),
        num_train_timesteps=mc.get("num_train_timesteps", 1000),
    ).to(device).eval()

    new_sd = {}
    for k, v in state_dict.items():
        for pre in ["generator.", "model.", "model_diff."]:
            if k.startswith(pre):
                k = k[len(pre):]
                break
        new_sd[k] = v
    for k in list(new_sd.keys()):
        if "conv_in" in k and k.startswith("unet.conv_in"):
            del new_sd[k]

    generator.load_state_dict(new_sd, strict=False)
    return generator, sr_model


def build_dataloader(cfg, batch_size, max_samples):
    mc = cfg.get("model", {})
    data_cfg = cfg["data"]
    ds = RefLMDBDataset(
        mode="val", data_dir=data_cfg["root"],
        patch_size=data_cfg.get("patch_size", 480),
        scale=data_cfg.get("scale", 10),
        ref_aug_strengths=data_cfg.get("ref_aug_strengths", [0.15, 0.15, 0.15, 0.03]),
        ref_aug_probs=data_cfg.get("ref_aug_probs", [0.5, 0.5, 0.5, 0.5]),
        ref_gray_prob=data_cfg.get("ref_gray_prob", 0.3),
        max_samples=(None, max_samples, None),
        sample_seed=42,
        lr_key=mc.get("lr_key", "lr"),
        hr_key=mc.get("hr_key", "hr"),
        ref_key=mc.get("ref_key", "ref"),
        augment=False, augment_ref=False,
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)


def compute_psnr(pred, gt):
    mse = F.mse_loss(pred, gt).item()
    return 10.0 * np.log10(1.0 / max(mse, 1e-10))


def tensor_to_image(tensor):
    """tensor [3,H,W] ∈ [-1,1] → PIL.Image"""
    arr = tensor.detach().cpu().permute(1, 2, 0).numpy()
    arr = (arr * 0.5 + 0.5) * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ══════════════════════════════════════════════════════════════
#  标准 benchmark（只比较最终采样结果）
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def run_sample(generator, lr, ref, hr, t_start, sample_steps, guidance_scale=0.0, t_stop=0):
    samples = generator.sample_log(
        lr, ref, steps=sample_steps, sr_model=generator.sr_model,
        hr=hr, t_start=t_start,
        guidance_scale=guidance_scale, t_stop=t_stop,
    )
    return compute_psnr(samples, hr)


def run_benchmark(args, cfg, generator, dataloader, t_start_list, device, guidance_scale=0.0):
    all_psnrs = defaultdict(list)

    pbar = tqdm(total=args.num_samples * len(t_start_list), desc=f"Benchmark (guid={guidance_scale})")
    for batch in dataloader:
        lr = batch["lr"].to(device)
        ref = batch["ref"].to(device)
        hr = batch["hr"].to(device)

        for t_start in t_start_list:
            psnr = run_sample(generator, lr, ref, hr, t_start, args.sample_steps,
                              guidance_scale=guidance_scale, t_stop=args.t_stop)
            all_psnrs[t_start].append(psnr)
            pbar.update(1)

        if len(all_psnrs[t_start_list[0]]) >= args.num_samples:
            break

    pbar.close()

    print("" + "=" * 70)
    gs_str = f", guidance_scale={guidance_scale}" if guidance_scale != 0.0 else ""
    print(f"  Better Start t_start 对比 — {args.sample_steps} 步 DDIM{gs_str}")
    print(f"  样本数: {args.num_samples}")
    print("=" * 70)
    print(f"{'t_start':>8}  {'PSNR mean':>10}  {'PSNR std':>10}  {' vs None':>10}")
    print("-" * 70)

    baseline_psnr = np.mean(all_psnrs[None]) if None in all_psnrs else None

    for t_start in t_start_list:
        psnrs = all_psnrs[t_start]
        mean_psnr = np.mean(psnrs)
        std_psnr = np.std(psnrs)
        label = "None" if t_start is None else str(t_start)
        delta = ""
        if baseline_psnr is not None and t_start is not None:
            delta = f"{mean_psnr - baseline_psnr:+.2f} dB"
        print(f"{label:>8}  {mean_psnr:>10.2f}  {std_psnr:>10.2f}  {delta:>10}")

    print("-" * 70)
    best_t = max(t_start_list, key=lambda t: np.mean(all_psnrs[t]))
    best_label = "None" if best_t is None else str(best_t)
    print(f"🏆 最佳 t_start: {best_label}  (PSNR = {np.mean(all_psnrs[best_t]):.2f} dB)")
    if baseline_psnr is not None:
        print(f"   vs 纯噪声: {np.mean(all_psnrs[best_t]) - baseline_psnr:+.2f} dB")


# ══════════════════════════════════════════════════════════════
#  轨迹对比：SR 路径 vs HR 路径
# ══════════════════════════════════════════════════════════════

@torch.no_grad()
def run_trajectory_comparison(generator, lr, ref, hr, t_start, sample_steps,
                               output_dir=None, sample_idx=0, save_interval=10):
    device = lr.device

    sr_pixel = generator.sr_model(lr, ref)
    sr_latent = generator.encode_latent(sr_pixel)
    sr_latent_cond = sr_latent.clone()

    hr_latent = generator.encode_latent(hr)
    noise = torch.randn_like(sr_latent)

    generator.noise_scheduler.set_timesteps(sample_steps, device=device)
    timesteps = generator.noise_scheduler.timesteps
    if t_start is not None:
        timesteps = [t for t in timesteps if t <= t_start]

    t_tensor = torch.full((1,), t_start, device=device, dtype=torch.long)
    x_sr = generator.noise_scheduler.add_noise(sr_latent, noise, t_tensor)
    x_hr = generator.noise_scheduler.add_noise(hr_latent, noise, t_tensor)

    sr_psnrs = []
    hr_psnrs = []
    cross_psnrs = []

    for step, t in enumerate(timesteps):
        t_t = torch.full((1,), int(t), device=device, dtype=torch.long)

        # SR 路径
        x_input_sr = generator._concat_sr_latent(x_sr, sr_latent_cond.detach())
        noise_pred_sr = generator.apply_model(x_input_sr, t_t, lr, ref)
        x_sr = generator.noise_scheduler.step(noise_pred_sr, t, x_sr).prev_sample
        pixel_sr = generator.decode_latent_eval(x_sr)

        # HR 路径
        x_input_hr = generator._concat_sr_latent(x_hr, sr_latent_cond.detach())
        null_ctx = torch.zeros(1, 77, generator.cross_attn_dim, device=device, dtype=torch.float32)
        eps_pred_hr = generator.unet(x_input_hr, t_t, encoder_hidden_states=null_ctx).sample
        x_hr = generator.noise_scheduler.step(eps_pred_hr, t, x_hr).prev_sample
        pixel_hr = generator.decode_latent_eval(x_hr)

        sr_psnrs.append((int(t), compute_psnr(pixel_sr, hr)))
        hr_psnrs.append((int(t), compute_psnr(pixel_hr, hr)))
        cross_psnrs.append((int(t), compute_psnr(pixel_sr, pixel_hr)))

        if output_dir and (step % save_interval == 0 or step == len(timesteps) - 1):
            for tag, pixel in [("SR", pixel_sr), ("HR", pixel_hr)]:
                img = tensor_to_image(pixel[0])
                os.makedirs(output_dir, exist_ok=True)
                img.save(os.path.join(output_dir,
                          f"sample{sample_idx}_tstart{t_start}_step{step:03d}_t{int(t)}_{tag}.png"))

    return sr_psnrs, hr_psnrs, cross_psnrs


def run_trajectory_benchmark(args, cfg, generator, dataloader, t_start_list, device):
    # 轨迹模式下上限 20 个样本（防止误传 100 跑两天）
    max_samples = min(args.num_samples, 50)

    if args.num_samples > max_samples:
        print(f"⚠️ 轨迹模式下 num_samples 上限为 {max_samples}，实际使用 {max_samples} 个样本")

    if args.save_images:
        if args.output_dir:
            img_dir = os.path.join(args.output_dir, "trajectory_images")
        else:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            img_dir = f"trajectory_images_{ts}"
    else:
        img_dir = None

    accumulated = {}
    for ts in t_start_list:
        if ts is None:
            continue
        accumulated[ts] = {"sr": None, "hr": None, "cross": None, "ts": None, "count": 0}

    # 总任务数：样本数 × t_start 个数
    total_tasks = max_samples * len([t for t in t_start_list if t is not None])
    pbar = tqdm(total=total_tasks, desc="轨迹对比")

    sample_count = 0
    for batch in dataloader:
        if sample_count >= max_samples:
            break

        lr = batch["lr"].to(device)
        ref = batch["ref"].to(device)
        hr = batch["hr"].to(device)

        for t_start in t_start_list:
            if t_start is None:
                continue

            sr_psnrs, hr_psnrs, cross_psnrs = run_trajectory_comparison(
                generator, lr, ref, hr, t_start, args.sample_steps,
                output_dir=img_dir, sample_idx=sample_count,
                save_interval=args.save_interval if args.save_images else 9999,
            )

            # 第一个 t_start 时初始化存储结构
            if accumulated[t_start]["sr"] is None:
                n_steps = len(sr_psnrs)
                accumulated[t_start]["sr"] = [[] for _ in range(n_steps)]
                accumulated[t_start]["hr"] = [[] for _ in range(n_steps)]
                accumulated[t_start]["cross"] = [[] for _ in range(n_steps)]
                accumulated[t_start]["ts"] = [t for t, _ in sr_psnrs]

            for i, (_, psnr) in enumerate(sr_psnrs):
                accumulated[t_start]["sr"][i].append(psnr)
            for i, (_, psnr) in enumerate(hr_psnrs):
                accumulated[t_start]["hr"][i].append(psnr)
            for i, (_, psnr) in enumerate(cross_psnrs):
                accumulated[t_start]["cross"][i].append(psnr)
            accumulated[t_start]["count"] += 1

            # 每个 t_start 完成后更新进度条
            pbar.update(1)

        sample_count += 1

    pbar.close()

    # ── 汇总输出 ──
    print("" + "=" * 100)
    print("  去噪轨迹对比：SR 路径(有adapter) vs HR 路径(无adapter)")
    print(f"  样本数: {sample_count}   |   步数: {args.sample_steps}")
    print("=" * 100)

    for t_start in sorted(accumulated.keys(), reverse=True):
        acc = accumulated[t_start]
        if acc["count"] == 0:
            continue
        n_steps = len(acc["sr"])
        ts_list = acc["ts"]

        print(f"{'─' * 100}")
        print(f"  t_start = {t_start}  ({n_steps} 步 DDIM)")
        print(f"{'─' * 100}")
        print(f"  {'Step':>5}  {'t':>5}  {'SR PSNR':>9}  {'HR PSNR':>9}"
              f"  {'Δ(SR-HR)':>9}  {'CROSS PSNR':>11}  {'趋势'}")
        print(f"  {'─' * 5}  {'─' * 5}  {'─' * 9}  {'─' * 9}  {'─' * 9}  {'─' * 11}  {'─' * 20}")

        prev_abs_delta = None
        for step_idx in range(n_steps):
            sr_mean = np.mean(acc["sr"][step_idx])
            hr_mean = np.mean(acc["hr"][step_idx])
            cross_mean = np.mean(acc["cross"][step_idx])
            delta = sr_mean - hr_mean
            t_val = ts_list[step_idx]

            trend = ""
            if prev_abs_delta is not None:
                if abs(delta) < prev_abs_delta - 0.05:
                    trend = "↑ 差距缩小 (收敛)"
                elif abs(delta) > prev_abs_delta + 0.05:
                    trend = "↓ 差距扩大 (发散)"
                else:
                    trend = "→ 持平"
            prev_abs_delta = abs(delta)

            if step_idx == 0 or step_idx == n_steps - 1 or step_idx % max(1, n_steps // 10) == 0:
                print(f"  {step_idx:>5}  {t_val:>5}  {sr_mean:>9.2f}  {hr_mean:>9.2f}  "
                      f"{delta:>+9.2f}  {cross_mean:>11.2f}  {trend}")

        init_delta = np.mean(acc["sr"][0]) - np.mean(acc["hr"][0])
        final_delta = np.mean(acc["sr"][-1]) - np.mean(acc["hr"][-1])
        init_cross = np.mean(acc["cross"][0])
        final_cross = np.mean(acc["cross"][-1])

        print(f"  📊 初始: Δ={init_delta:+.2f} dB, CROSS={init_cross:.2f} dB")
        print(f"     最终: Δ={final_delta:+.2f} dB, CROSS={final_cross:.2f} dB")

        if abs(final_delta) < abs(init_delta):
            print(f"     ✅ 轨迹收敛（|Δ| 从 {abs(init_delta):.2f} → {abs(final_delta):.2f}）")
        else:
            print(f"     ⚠️ 轨迹发散（|Δ| 从 {abs(init_delta):.2f} → {abs(final_delta):.2f}）")

        if final_cross > 25:
            print(f"     🔍 最终 SR 与 HR 几乎一致 (CROSS PSNR > 25 dB)")
        elif final_cross > 18:
            print(f"     🔎 最终 SR 与 HR 较接近 (CROSS PSNR > 18 dB)")
        else:
            print(f"     🔴 最终 SR 与 HR 差异较大")


# ══════════════════════════════════════════════════════════════
#  Guidance Scale 扫描
# ══════════════════════════════════════════════════════════════

def run_guidance_scan(args, cfg, generator, dataloader, guidance_list, device):
    t_start_list = []
    for s in args.t_start_list:
        if s.lower() == "none" or s == "0":
            t_start_list.append(None)
        else:
            t_start_list.append(int(s))

    all = {}
    for gs in guidance_list:
        print(f"{'='*60}")
        print(f"  🔬 测试 guidance_scale = {gs}")
        print(f"{'='*60}")
        psnrs = {}
        pbar = tqdm(total=args.num_samples * len(t_start_list), desc=f"guid={gs}")
        for batch in dataloader:
            lr = batch["lr"].to(device)
            ref = batch["ref"].to(device)
            hr = batch["hr"].to(device)
            for t_start in t_start_list:
                psnr = run_sample(generator, lr, ref, hr, t_start, args.sample_steps,
                                  guidance_scale=gs, t_stop=args.t_stop)
                psnrs.setdefault(t_start, []).append(psnr)
                pbar.update(1)
            if t_start_list and len(psnrs.get(t_start_list[0], [])) >= args.num_samples:
                break
        pbar.close()
        all[gs] = psnrs

    print("" + "=" * 80)
    print(f"  MSE Guidance 扫描 — {args.sample_steps} 步 DDIM (t_stop={args.t_stop})")
    print(f"  样本数: {args.num_samples}")
    print("=" * 80)
    header = f"{'guid':>6}  "
    for ts in t_start_list:
        label = "None" if ts is None else f"t={ts}"
        header += f"{label:>10}  "
    print(header)
    print("-" * 80)

    for gs in guidance_list:
        line = f"{gs:>6}  "
        for ts in t_start_list:
            m = np.mean(all[gs][ts])
            line += f"{m:>10.2f}  "
        print(line)

    best_combo = None
    best_psnr = -float("inf")
    for gs, psnrs in all.items():
        for ts, vals in psnrs.items():
            m = np.mean(vals)
            if m > best_psnr:
                best_psnr = m
                best_combo = (gs, ts)
    print(f"🏆 最佳组合: guidance_scale={best_combo[0]}, t_start={best_combo[1]}  "
          f"(PSNR={best_psnr:.2f} dB)")


# ══════════════════════════════════════════════════════════════
#  main
# ══════════════════════════════════════════════════════════════

def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    t_start_list = []
    for s in args.t_start_list:
        if s.lower() == "none" or s == "0":
            t_start_list.append(None)
        else:
            t_start_list.append(int(s))

    print(f"🔧 加载模型 ({args.ckpt})...")
    generator, _ = load_refgan(cfg, args.ckpt, device)

    print(f"📦 加载数据...")
    dataloader = build_dataloader(cfg, batch_size=1, max_samples=args.num_samples)

    if args.guidance_scale_list is not None:
        run_guidance_scan(args, cfg, generator, dataloader,
                          args.guidance_scale_list, device)
    elif args.compare_trajectory:
        run_trajectory_benchmark(args, cfg, generator, dataloader,
                                 t_start_list, device)
    else:
        run_benchmark(args, cfg, generator, dataloader,
                      t_start_list, device)

    print(f"📋 配置:")
    print(f"  t_start_list: {['None' if t is None else str(t) for t in t_start_list]}")
    print(f"  sample_steps: {args.sample_steps}")
    print(f"  num_samples: {args.num_samples}")


if __name__ == "__main__":
    main()
