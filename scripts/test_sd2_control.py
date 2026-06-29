#!/usr/bin/env python
"""
SD2ControlLDM 测试/推理脚本（RefDiffRWKV + SD2 UNet）
用法:
    python scripts/test_sd2_control.py --config configs/test_config.yaml
    python scripts/test_sd2_control.py --config configs/test_config.yaml --ckpt checkpoints/.../best.ckpt
    python scripts/test_sd2_control.py --config configs/test_config.yaml --save_video
"""

import sys
import argparse
import yaml
import os
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from tqdm import tqdm

from RefRWKV.models.RefDiffRWKV.sd2_control_ldm import SD2ControlLDM
from RefRWKV.RefSR_data.RefSR_dataset import RefLMDBDataset

torch.set_float32_matmul_precision("high")


# ============================================================
# 度量计算
# ============================================================
@torch.no_grad()
def compute_metrics(sr, hr, iqa, fr_metrics, device):
    """计算一组图像的平均 IQA 指标。
    Args:
        sr: [B, 3, H, W] range [0, 1]
        hr: [B, 3, H, W] range [0, 1]
    Returns:
        dict: {metric_name: mean_value}
    """
    B = sr.shape[0]
    accum = {m: 0.0 for m in fr_metrics}
    for i in range(B):
        r = iqa.evaluate_single(
            sr[i].cpu().float().permute(1, 2, 0).numpy(),
            hr[i].cpu().float().permute(1, 2, 0).numpy(),
        )
        for k in accum:
            accum[k] += r.get(k, 0.0)
    return {k: v / B for k, v in accum.items()}


# ============================================================
# 配置加载
# ============================================================
def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_weights(model, ckpt_path: str):
    """从 Lightning checkpoint 加载模型权重。"""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    new_state_dict = {}
    for k, v in state_dict.items():
        for pre in ["model.", "model_sr.", "model_diff.", "model_enhance."]:
            if k.startswith(pre):
                k = k[len(pre):]
                break
        new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    if missing:
        print(f"⚠️  缺失键 ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"⚠️  多余键 ({len(unexpected)}): {unexpected[:5]}...")
    print(f"✅ 从 {ckpt_path} 加载权重完成")


def build_test_dataloader(cfg: dict):
    """返回 test_loader。"""
    data_cfg = cfg["data"]
    mc = cfg.get("model", {})

    max_samples_tuple = (
        None,
        None,
        data_cfg.get("max_samples_test"),
    )

    ds = RefLMDBDataset(
        mode="test",
        data_dir=data_cfg["root"],
        patch_size=data_cfg.get("patch_size", 480),
        scale=data_cfg.get("scale", 10),
        ref_aug_strengths=[0.0, 0.0, 0.0, 0.0],
        ref_aug_probs=[0.0, 0.0, 0.0, 0.0],
        ref_gray_prob=0.0,
        max_samples=max_samples_tuple,
        sample_seed=42,
        augment=False,
        augment_ref=False,
        lr_key=mc.get("lr_key", "lr"),
        hr_key=mc.get("hr_key", "hr"),
        ref_key=mc.get("ref_key", "ref"),
    )

    loader = DataLoader(
        ds,
        batch_size=data_cfg.get("val_batch_size", 1),
        shuffle=False,
        num_workers=data_cfg.get("val_num_workers", 0),
        pin_memory=True,
    )
    return loader


def build_model(cfg: dict, device) -> SD2ControlLDM:
    """根据配置构建 SD2ControlLDM。"""
    mc = cfg.get("model", {})

    return SD2ControlLDM(
        lr_key=mc.get("lr_key", "lr"),
        ref_key=mc.get("ref_key", "ref"),
        hr_key=mc.get("hr_key", "hr"),
        sd_model_path=mc["sd_model_path"],
        use_lora=mc.get("use_lora", True),
        lora_rank=mc.get("lora_rank", 4),
        lora_target_modules=mc.get("lora_target_modules", None),
        sd_locked=mc.get("sd_locked", True),
        patch_size=mc.get("patch_size", 4),
        embed_dim=mc.get("embed_dim", 384),
        upsample_mode=mc.get("upsample_mode", "bilinear"),
        use_semantic=mc.get("use_semantic", False),
        dinov2_model_name=mc.get("dinov2_model_name", "facebook/dinov2-base"),
        cfg_drop_prob=mc.get("cfg_drop_prob", 0.1),
        learning_rate=mc.get("learning_rate", 1e-4),
        num_train_timesteps=mc.get("num_train_timesteps", 1000),
        beta_start=mc.get("beta_start", 0.00085),
        beta_end=mc.get("beta_end", 0.012),
        beta_schedule=mc.get("beta_schedule", "scaled_linear"),
        prediction_type=mc.get("prediction_type", "epsilon"),
        l_simple_weight=mc.get("l_simple_weight", 1.0),
        weight_decay=mc.get("weight_decay", 1e-3),
        sample_steps=mc.get("sample_steps", 50),
        fr_metrics=mc.get("fr_metrics", None),
        iqa_device=mc.get("iqa_device", "cuda"),
        debug_nan=mc.get("debug_nan", False),
    ).to(device)


# ============================================================
# 单张保存
# ============================================================
def save_tensor(tensor, path):
    """保存 tensor [C, H, W] range [-1, 1] 为 PNG。"""
    img = (tensor.clamp(-1, 1) + 1) / 2
    save_image(img, path)


def build_comparison(lr, ref, sr, hr, save_to):
    """拼接对比图: lr(放大) | ref | sr | hr。"""
    lr, ref, sr, hr = lr.cpu(), ref.cpu(), sr.cpu(), hr.cpu()
    
    lr_up = F.interpolate(
        lr.unsqueeze(0), size=ref.shape[-2:], mode="bilinear", align_corners=False
    ).squeeze(0)

    comp = torch.cat([lr_up, ref, sr, hr], dim=-1)
    save_tensor(comp, save_to)


# ============================================================
# 主逻辑
# ============================================================
def test(cfg: dict, ckpt_path: str):
    tc = cfg.get("test", {})
    mc = cfg.get("model", {})

    # ── 输出目录 ──
    ckpt_name = Path(ckpt_path).stem
    output_dir = Path(tc.get("output_dir", "outputs/test")) / ckpt_name
    output_dir.mkdir(parents=True, exist_ok=True)

    sr_dir = output_dir / "sr"
    comp_dir = output_dir / "comparison"
    sr_dir.mkdir(exist_ok=True)
    comp_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 模型 ──
    print("Loading model...")
    model = build_model(cfg, device)
    load_weights(model, ckpt_path)
    model.eval()

    # ── 数据 ──
    print("Loading test data...")
    loader = build_test_dataloader(cfg)

    # ── IQA（可选）──
    compute_iqa = tc.get("compute_iqa", True)
    fr_metrics = mc.get("fr_metrics", ["psnr", "ssim", "lpips", "dists"])
    if compute_iqa and model.iqa is not None:
        iqa_accum = {m: 0.0 for m in fr_metrics}
        iqa_count = 0
    else:
        iqa_accum, iqa_count = None, 0

    # ── 推理参数 ──
    sample_steps = tc.get("sample_steps", mc.get("sample_steps", 50))
    base_seed = tc.get("seed", 42)

    # ── 打印摘要 ──
    print(f"{'=' * 60}")
    print(f"  SD2ControlLDM 测试")
    print(f"  Checkpoint   : {ckpt_path}")
    print(f"  测试样本数   : {len(loader.dataset)}")
    print(f"  采样步数     : {sample_steps}")
    print(f"  输出目录     : {output_dir}")
    print(f"  计算 IQA     : {compute_iqa and model.iqa is not None}")
    print(f"{'=' * 60}")

    # ── 推理循环 ──
    all_metrics = []
    start_time = time.time()

    for batch_idx, batch in enumerate(
        tqdm(loader, desc="Testing", unit="sample")
    ):
        lr = batch["lr"].float().to(device)
        ref = batch["ref"].float().to(device)
        hr = batch["hr"].float().to(device)

        B = lr.shape[0]
        sr_list = []

        for i in range(B):
            with torch.no_grad():
                sr = model.inference(
                    lr[i : i + 1],
                    ref[i : i + 1],
                    steps=sample_steps,
                    seed=base_seed + batch_idx * B + i,
                )  # [1, 3, H, W] range [0, 1]

            sr = sr.squeeze(0)  # [3, H, W]
            sr_list.append(sr)

        sr_batch = torch.stack(sr_list)  # [B, 3, H, W]

        # ── 保存单张 ──
        for i in range(B):
            idx = batch_idx * B + i
            save_tensor(sr_list[i] * 2 - 1, sr_dir / f"{idx:06d}.png")

        # ── 保存对比图 ──
        if tc.get("save_comparison", True):
            for i in range(B):
                idx = batch_idx * B + i
                build_comparison(
                    lr[i].cpu(),           # [-1, 1]
                    ref[i].cpu(),          # [-1, 1]
                    (sr_list[i] * 2 - 1).cpu(),    # [0, 1] → [-1, 1]
                    hr[i].cpu(),           # [-1, 1]
                    comp_dir / f"{idx:06d}.png",
                )

        # ── IQA ──
        if iqa_accum is not None:
            hr_norm = (hr + 1) / 2  # [-1, 1] → [0, 1]
            m = compute_metrics(sr_batch, hr_norm, model.iqa, fr_metrics, device)
            for k, v in m.items():
                iqa_accum[k] += v
            iqa_count += 1
            all_metrics.append(m)

    elapsed = time.time() - start_time

    # ── 打印结果 ──
    print(f"{'=' * 60}")
    print(f"  ✅ 测试完成 ({elapsed:.1f}s, {len(loader.dataset)} 张)")
    print(f"  平均速度: {len(loader.dataset) / elapsed:.2f} 张/秒")
    print(f"  输出目录: {output_dir}")
    print(f"{'=' * 60}")

    # ── 打印 IQA 汇总 ──
    if iqa_accum is not None and iqa_count > 0:
        print(f"📊  IQA 指标汇总 ({iqa_count} batches):")
        print(f"  {'Metric':<10} {'Mean':>8}")
        print(f"  {'-' * 20}")
        for k in fr_metrics:
            mean_val = iqa_accum[k] / iqa_count
            print(f"  {k:<10} {mean_val:>8.4f}")

        # 写入文件
        with open(output_dir / "metrics.txt", "w") as f:
            f.write(f"Checkpoint: {ckpt_path}")
            f.write(f"Samples: {len(loader.dataset)}")
            f.write(f"Sample steps: {sample_steps}")
            f.write(f"{'='*40}")
            for k in fr_metrics:
                f.write(f"{k}: {iqa_accum[k] / iqa_count:.6f}")

    return output_dir


# ============================================================
# 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="SD2ControlLDM 测试/推理")
    parser.add_argument("--config", type=str, required=True, help="测试配置文件路径")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="checkpoint 路径。留空则自动查找 config 中指定的，或使用 best.ckpt",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ── 确定 checkpoint ──
    if args.ckpt:
        ckpt_path = args.ckpt
    elif cfg.get("test", {}).get("checkpoint"):
        ckpt_path = cfg["test"]["checkpoint"]
    else:
        # 自动找
        exp_name = cfg.get("output", {}).get("experiment_name", "sd2_control_ldm")
        ckpt_dir = Path(cfg.get("output", {}).get("checkpoint_dir", "checkpoints"))
        ckpt_dir = ckpt_dir / exp_name
        candidates = sorted(ckpt_dir.glob("*.ckpt"))
        if not candidates:
            raise FileNotFoundError(f"未找到 .ckpt 文件于 {ckpt_dir}")
        ckpt_path = str(candidates[-1])  # 取最新的
        print(f"🔍 自动选择: {ckpt_path}")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint 不存在: {ckpt_path}")

    test(cfg, ckpt_path)


if __name__ == "__main__":
    main()
