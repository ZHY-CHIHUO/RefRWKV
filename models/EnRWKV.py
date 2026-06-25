# Copyright (c) Shanghai AI Lab. All rights reserved.
import math, os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
import sys
from pathlib import Path

# 添加项目根目录到 sys.path
root_dir = str(Path(__file__).parent.parent)  # 从 models/ 向上到 RefRWKV/
sys.path.insert(0, root_dir)
from models.RefSRWKV import Block


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class EnRWKV(nn.Module):
    """
    基于视觉RWKV的U-Net图像恢复模型。

    整体结构：
        - 浅层特征提取（patch_embed）
        - 四级编码器，每级由多个 VRWKV Block 组成，逐级下采样
        - 底层 latent 处理
        - 四级解码器，通过跳跃连接与编码器特征融合，逐级上采样
        - 细化模块（refinement）
        - 输出层 + 全局残差连接
    """

    def __init__(
        self,
        inp_channels=3,  # 输入图像通道数
        out_channels=3,  # 输出图像通道数
        dim=48,  # 第一层特征通道数（基础宽度）
        num_blocks=[4, 6, 6, 8],  # 各编码/解码级的 Block 数量，依次为 level1 ~ level4
        num_refinement_blocks=4,  # 解码器末尾额外添加的细化 Block 数量
        loss_fun=nn.L1Loss(),  # 训练损失函数，默认 L1
    ):
        super(EnRWKV, self).__init__()

        # ===== 浅层特征提取 =====
        # 3x3卷积，将输入图像映射到 dim 维度，保持分辨率
        self.patch_embed = nn.Conv2d(
            inp_channels, dim, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.loss_fun = loss_fun

        # ===== 编码器（Encoder）=====
        # Level 1: 输入 dim 通道，输出 dim 通道，分辨率不变
        self.encoder_level1 = nn.Sequential(
            *[Block(n_embd=dim) for _ in range(num_blocks[0])]
        )

        # 下采样 Level1 → Level2 : 分辨率减半，通道数变为 dim*2
        self.down1_2 = Downsample(dim)  # 输入dim，输出dim*2
        self.encoder_level2 = nn.Sequential(
            *[Block(n_embd=int(dim * 2**1)) for _ in range(num_blocks[1])]
        )

        # 下采样 Level2 → Level3 : 通道数变为 dim*4
        self.down2_3 = Downsample(int(dim * 2**1))
        self.encoder_level3 = nn.Sequential(
            *[Block(n_embd=int(dim * 2**2)) for _ in range(num_blocks[2])]
        )

        # 下采样 Level3 → Level4 (latent): 通道数变为 dim*8
        self.down3_4 = Downsample(int(dim * 2**2))
        self.latent = nn.Sequential(
            *[Block(n_embd=int(dim * 2**3)) for _ in range(num_blocks[3])]
        )

        # ===== 解码器（Decoder）=====
        # 上采样 Level4 → Level3 : 分辨率加倍，通道数变为 dim*4
        self.up4_3 = Upsample(int(dim * 2**3))  # 输入dim*8，输出dim*4
        # 将拼接后的 dim*8 通道压缩为 dim*4
        self.reduce_chan_level3 = nn.Conv2d(
            int(dim * 2**3), int(dim * 2**2), kernel_size=1, bias=True
        )
        self.decoder_level3 = nn.Sequential(
            *[Block(n_embd=int(dim * 2**2)) for _ in range(num_blocks[2])]
        )

        # 上采样 Level3 → Level2 : 输出 dim*2
        self.up3_2 = Upsample(int(dim * 2**2))  # 输入dim*4，输出dim*2
        self.reduce_chan_level2 = nn.Conv2d(
            int(dim * 2**2), int(dim * 2**1), kernel_size=1, bias=True
        )
        self.decoder_level2 = nn.Sequential(
            *[Block(n_embd=int(dim * 2**1)) for _ in range(num_blocks[1])]
        )

        # 上采样 Level2 → Level1 : 输出 dim（与 encoder_level1 通道数相同）
        self.up2_1 = Upsample(int(dim * 2**1))  # 输入dim*2，输出dim
        # 拼接后通道数为 dim*2，直接送入 Block，不再用 1x1 降维
        self.decoder_level1 = nn.Sequential(
            *[Block(n_embd=int(dim * 2**1)) for _ in range(num_blocks[0])]
        )

        # ===== 细化模块 =====
        self.refinement = nn.Sequential(
            *[Block(n_embd=int(dim * 2**1)) for _ in range(num_refinement_blocks)]
        )

        # ===== 输出层 =====
        # 将 dim*2 特征映射到输出通道，并加上输入形成全局残差
        self.output = nn.Conv2d(
            int(dim * 2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=True
        )

        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def forward(self, inp_img, label=None):
        """
        前向传播。

        参数:
            inp_img: 输入图像，shape (B, C, H, W)
            label:   Ground truth 图像，若不为 None 则直接返回损失值，否则返回增强结果

        返回:
            若 label 不为 None，返回损失标量；
            否则返回恢复图像，shape (B, out_channels, H, W)
        """

        # ---------- 编码 ----------
        inp_enc_level1 = self.patch_embed(inp_img)  # (B, dim, H, W)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)  # (B, dim, H, W)

        inp_enc_level2 = self.down1_2(out_enc_level1)  # (B, dim*2, H/2, W/2)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)  # (B, dim*2, H/2, W/2)

        inp_enc_level3 = self.down2_3(out_enc_level2)  # (B, dim*4, H/4, W/4)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)  # (B, dim*4, H/4, W/4)

        inp_enc_level4 = self.down3_4(out_enc_level3)  # (B, dim*8, H/8, W/8)
        latent = self.latent(inp_enc_level4)  # (B, dim*8, H/8, W/8)

        # ---------- 解码（带跳跃连接）----------
        # Level4 -> Level3
        inp_dec_level3 = self.up4_3(latent)  # (B, dim*4, H/4, W/4)
        inp_dec_level3 = torch.cat(
            [inp_dec_level3, out_enc_level3], dim=1
        )  # (B, dim*8, H/4, W/4)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)  # (B, dim*4, H/4, W/4)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)  # (B, dim*4, H/4, W/4)

        # Level3 -> Level2
        inp_dec_level2 = self.up3_2(out_dec_level3)  # (B, dim*2, H/2, W/2)
        inp_dec_level2 = torch.cat(
            [inp_dec_level2, out_enc_level2], dim=1
        )  # (B, dim*4, H/2, W/2)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)  # (B, dim*2, H/2, W/2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)  # (B, dim*2, H/2, W/2)

        # Level2 -> Level1
        inp_dec_level1 = self.up2_1(out_dec_level2)  # (B, dim, H, W)
        inp_dec_level1 = torch.cat(
            [inp_dec_level1, out_enc_level1], dim=1
        )  # (B, dim*2, H, W)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)  # (B, dim*2, H, W)

        # 细化
        out_dec_level1 = self.refinement(out_dec_level1)  # (B, dim*2, H, W)

        # 输出 + 全局残差连接（假设输入与输出空间对齐，例如去噪、增强等任务）
        out = self.output(out_dec_level1) + inp_img  # (B, out_channels, H, W)

        # 根据是否提供 label 决定返回结果或损失
        if label is None:
            return out
        else:
            return self.loss_fun(out, label)

class LitEnRWKV(pl.LightningModule):
    def __init__(
        self,
        model_enhance: nn.Module,
        model_diff: nn.Module,
        num_timesteps: int = 1000,
        t_threshold: int = 250,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        warmup_epochs: int = 5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model_enhance", "model_diff"])

        self.model_enhance = model_enhance
        self.model_diff = model_diff
        self.num_timesteps = num_timesteps
        self.t_threshold = t_threshold
        self.lr = learning_rate
        self.wd = weight_decay
        self.warmup_epochs = warmup_epochs

        # 冻结扩散模型
        for p in self.model_diff.parameters():
            p.requires_grad = False
        self.model_diff.eval()

    def _add_noise(self, x0, noise, t):
        s = 0.008
        T = self.num_timesteps
        alpha_bar = torch.cos(((t.float() / T + s) / (1 + s)) * math.pi / 2) ** 2
        alpha_bar = alpha_bar.view(-1, 1, 1, 1)
        x_t = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise
        return x_t, noise, alpha_bar

    def _step(self, batch, stage):
        lr, hr, ref = batch
        B = hr.shape[0]
        device = hr.device

        t = torch.randint(1, self.t_threshold + 1, (B,), device=device)
        noise = torch.randn_like(hr)
        x_t, noise, alpha_bar = self._add_noise(hr, noise, t)

        with torch.no_grad():
            pred_noise = self.model_diff(x_t, t, lr, ref)
            pred_noise = torch.clamp(pred_noise, -5.0, 5.0)
            alpha_bar_t = alpha_bar.view(-1, 1, 1, 1)
            pred_x0 = (
                x_t - torch.sqrt(1 - alpha_bar_t) * pred_noise
            ) / torch.sqrt(alpha_bar_t + 1e-8)
            pred_x0 = torch.clamp(pred_x0, -1, 1)

        refined = self.model_enhance(pred_x0, label=None)
        loss = F.l1_loss(refined, hr)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd)
        total_epochs = self.trainer.max_epochs if self.trainer else 200

        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return float(epoch + 1) / float(max(1, self.warmup_epochs))
            progress = float(epoch - self.warmup_epochs) / float(
                max(1, total_epochs - self.warmup_epochs)
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }

    # ----- 强制扩散模型保持 eval 模式 -----
    def on_train_start(self):
        self.model_diff.eval()
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"✅ LitEnRWKV 训练开始 | 总参数: {total / 1e6:.2f}M | 可训练: {trainable / 1e6:.2f}M")

    def on_validation_start(self):
        self.model_diff.eval()

    def on_test_start(self):
        self.model_diff.eval()


