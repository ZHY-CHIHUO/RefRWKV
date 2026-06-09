#!/usr/bin/env python
# test.py
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
import os

# 导入模型和数据集
from model import RefDiffRWKV, RefDiffRWKV_PL  # 根据你的目录结构调整
from RWKV.RefSR_data.RefSR_dataset import RefPNGDataset
from RWKV.evaluation.eval_pyiqa import IQAEngine  # 你已有的评估引擎


# ====================== DDIM 采样器 ======================
@torch.no_grad()
def ddim_sample(model, lr, ref, shape, ddim_steps=50, eta=0.0):
    """
    DDIM 采样，生成 HR 图像。
    model: RefDiffRWKV 实例（必须已加载权重）
    lr: 低分辨率图像 (B, C, H_lr, W_lr)
    ref: 参考图像 (B, C, H, W)
    shape: 输出图像的形状 (B, C, H, W)
    ddim_steps: 采样步数（默认 50）
    eta: DDIM 噪声系数，0 为确定性采样
    """
    batch_size = shape[0]
    device = next(model.parameters()).device
    T = 1000  # 训练时的总时间步数（需与训练配置一致）

    # 生成时间步序列（均匀选取）
    times = torch.linspace(T, 0, ddim_steps + 1, device=device).long()
    times = times[:-1]  # 去掉最后的 0 步
    times_next = torch.cat([times[1:], torch.tensor([0], device=device)])

    # 初始随机噪声
    img = torch.randn(shape, device=device)

    for i in tqdm(range(ddim_steps), desc="DDIM sampling"):
        t = times[i].repeat(batch_size)  # 当前时间步
        t_next = times_next[i]  # 下一时间步

        # 预测噪声
        pred_noise = model(img, t, lr, ref)

        # 计算 alpha_bar
        s = 0.008
        alpha_bar_t = torch.cos(((t.float() / T + s) / (1 + s)) * np.pi / 2) ** 2
        alpha_bar_t_next = (
            torch.cos(((t_next.float() / T + s) / (1 + s)) * np.pi / 2) ** 2
        )

        # 预测 x0
        pred_x0 = (
            img - torch.sqrt(1 - alpha_bar_t.view(-1, 1, 1, 1)) * pred_noise
        ) / torch.sqrt(alpha_bar_t.view(-1, 1, 1, 1))

        # 方向指向下一时间步
        if eta > 0:
            sigma = (
                eta
                * torch.sqrt((1 - alpha_bar_t_next) / (1 - alpha_bar_t))
                * torch.sqrt(1 - alpha_bar_t / alpha_bar_t_next)
            )
            noise = torch.randn_like(img)
            img = (
                torch.sqrt(alpha_bar_t_next.view(-1, 1, 1, 1)) * pred_x0
                + torch.sqrt(1 - alpha_bar_t_next.view(-1, 1, 1, 1) - sigma**2)
                * pred_noise
                + sigma * noise
            )
        else:
            # eta=0 确定性采样
            img = (
                torch.sqrt(alpha_bar_t_next.view(-1, 1, 1, 1)) * pred_x0
                + torch.sqrt(1 - alpha_bar_t_next.view(-1, 1, 1, 1)) * pred_noise
            )

    return img


def main():
    # ====================== 配置 ======================
    # 模型权重路径（请根据实际情况修改）
    model_ckpt = "checkpoints/refdiff-epoch=XXXX-val_loss=XXXXX.ckpt"
    data_root = r"/home/zhy/PROJECT/RWKV/RefSR_data/ALL_2"  # 数据集根目录
    output_dir = "test_results/images"
    os.makedirs(output_dir, exist_ok=True)

    # 采样参数
    ddim_steps = 50
    batch_size = 1  # 测试时通常 batch=1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型参数（需与训练时一致）
    model_args = dict(
        img_size=480,  # 模型内部其实不关键，但保持一致
        patch_size=4,
        embed_dim=64,
        enc_blocks=[4, 6, 6],
        dec_blocks=[6, 6, 4],
        latent_blocks=8,
        drop_path_rate=0.0,
        upsample_mode="cnn",
        channels=3,
    )

    # 数据集参数
    dataset_kwargs = dict(
        data_dir=data_root,
        mode="test",  # 使用测试集
        patch_size=160,  # 使用全图
        augment=False,
        max_samples=(None, None, None),
        sample_seed=42,
    )

    # ====================== 加载模型 ======================
    print("Loading model...")
    base_model = RefDiffRWKV(**model_args)
    pl_model = RefDiffRWKV_PL.load_from_checkpoint(
        model_ckpt, model=base_model, strict=False
    )
    model = pl_model.model
    model.eval()
    model.to(device)
    print("Model loaded successfully.")

    # ====================== 加载测试数据集 ======================
    print("Loading test dataset...")
    test_dataset = RefPNGDataset(**dataset_kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True
    )
    print(f"Number of test samples: {len(test_dataset)}")

    # ====================== 推理循环 ======================
    all_pred_paths = []
    gt_paths = []

    for idx, batch in enumerate(tqdm(test_loader, desc="Processing")):
        lr, hr, ref = batch  # 注意：数据集返回顺序 (LR, HR, Ref)
        lr = lr.to(device)
        hr = hr.to(device)
        ref = ref.to(device)
        B, C, H, W = hr.shape

        # 上采样 LR 到与 HR 相同尺寸
        lr_up = F.interpolate(lr, size=(H, W), mode="bilinear", align_corners=False)

        # DDIM 采样生成 HR 图像
        generated = ddim_sample(
            model, lr_up, ref, shape=(B, C, H, W), ddim_steps=ddim_steps, eta=0.0
        )

        # 后处理：将 [-1,1] 或 [0,1] 归一化到 [0,255] 并保存
        # 注意：训练时 HR 输入可能是 [0,1] 范围，输出也在此范围
        generated = generated.clamp(0, 1).squeeze(0).cpu()
        gen_img = (generated.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        out_path = os.path.join(output_dir, f"{idx:05d}.png")
        Image.fromarray(gen_img).save(out_path)
        all_pred_paths.append(out_path)

        # 保存对应的 GT 图像（用于评估）
        hr_img = (hr.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        gt_path = os.path.join(output_dir, f"{idx:05d}_gt.png")
        Image.fromarray(hr_img).save(gt_path)
        gt_paths.append(gt_path)

    # ====================== 评估 ======================
    print("\nRunning IQA evaluation...")
    engine = IQAEngine(device="cuda", verbose=True)
    # 将所有预测图像与 GT 图像比较
    # 方法1：批量文件夹评估（需将 pred 和 gt 分别放到两个文件夹）
    pred_dir = output_dir
    gt_dir = "/home/zhy/PROJECT/RWKV/RefSR_data/ALL_2/test/HR"

    engine = IQAEngine(device='cuda', verbose=True)
    results = engine.evaluate_folder(
        pred_dir=pred_dir,
        gt_dir=gt_dir,
        ntest=None,           # 评估全部图像，或指定数量
        out_path=output_dir   # 结果保存目录
    )

    print("Evaluation Results:")
    for k, v in results.items():
        print(f"{k}: {v:.5f}")


if __name__ == "__main__":
    main()
