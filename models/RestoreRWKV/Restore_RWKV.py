# Copyright (c) Shanghai AI Lab. All rights reserved.
import math, os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
from torch.utils.cpp_extension import load

wkv_cuda = load(
    name="bi_wkv",
    sources=["./cuda/bi_wkv.cpp", "./cuda/bi_wkv_kernel.cu"],
    verbose=True,
    extra_cuda_cflags=[
        "-res-usage",
        "--maxrregcount 60",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        "-gencode arch=compute_120,code=sm_120",
    ],
)


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, u, k, v):
        half_mode = w.dtype == torch.half
        bf_mode = w.dtype == torch.bfloat16
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = wkv_cuda.bi_wkv_forward(w, u, k, v)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        w, u, k, v = ctx.saved_tensors
        half_mode = w.dtype == torch.half
        bf_mode = w.dtype == torch.bfloat16
        gw, gu, gk, gv = wkv_cuda.bi_wkv_backward(
            w.float().contiguous(),
            u.float().contiguous(),
            k.float().contiguous(),
            v.float().contiguous(),
            gy.float().contiguous(),
        )
        if half_mode:
            return (gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            return (gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            return (gw, gu, gk, gv)


def RUN_CUDA(w, u, k, v):
    return WKV.apply(w.cuda(), u.cuda(), k.cuda(), v.cuda())


class OmniShift(nn.Module):
    def __init__(self, dim):
        super(OmniShift, self).__init__()
        # Define the layers for training
        self.conv1x1 = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=1, groups=dim, bias=False
        )
        self.conv3x3 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=1,
            groups=dim,
            bias=False,
        )
        self.conv5x5 = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=5,
            padding=2,
            groups=dim,
            bias=False,
        )
        self.alpha = nn.Parameter(torch.randn(4), requires_grad=True)

        # Define the layers for testing
        self.conv5x5_reparam = nn.Conv2d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=5,
            padding=2,
            groups=dim,
            bias=False,
        )
        self.repram_flag = True

    def forward_train(self, x):
        out1x1 = self.conv1x1(x)
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x)
        # import pdb
        # pdb.set_trace()

        out = (
            self.alpha[0] * x
            + self.alpha[1] * out1x1
            + self.alpha[2] * out3x3
            + self.alpha[3] * out5x5
        )
        return out

    def reparam_5x5(self):
        # Combine the parameters of conv1x1, conv3x3, and conv5x5 to form a single 5x5 depth-wise convolution

        padded_weight_1x1 = F.pad(self.conv1x1.weight, (2, 2, 2, 2))
        padded_weight_3x3 = F.pad(self.conv3x3.weight, (1, 1, 1, 1))

        identity_weight = F.pad(torch.ones_like(self.conv1x1.weight), (2, 2, 2, 2))

        combined_weight = (
            self.alpha[0] * identity_weight
            + self.alpha[1] * padded_weight_1x1
            + self.alpha[2] * padded_weight_3x3
            + self.alpha[3] * self.conv5x5.weight
        )

        device = self.conv5x5_reparam.weight.device

        combined_weight = combined_weight.to(device)

        self.conv5x5_reparam.weight = nn.Parameter(combined_weight)

    def forward(self, x):

        if self.training:
            self.repram_flag = True
            out = self.forward_train(x)
        elif self.training == False and self.repram_flag == True:
            self.reparam_5x5()
            self.repram_flag = False
            out = self.conv5x5_reparam(x)
        elif self.training == False and self.repram_flag == False:
            out = self.conv5x5_reparam(x)

        return out


class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd

        self.recurrence = 2

        self.omni_shift = OmniShift(dim=n_embd)

        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        self.output = nn.Linear(attn_sz, n_embd, bias=False)

        with torch.no_grad():
            self.spatial_decay = nn.Parameter(
                torch.randn((self.recurrence, self.n_embd))
            )
            self.spatial_first = nn.Parameter(
                torch.randn((self.recurrence, self.n_embd))
            )

    def jit_func(self, x, resolution):
        # Mix x with the previous timestep to produce xk, xv, xr

        h, w = resolution

        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.omni_shift(x)
        x = rearrange(x, "b c h w -> b (h w) c")

        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        sr = torch.sigmoid(r)

        return sr, k, v

    def forward(self, x, resolution):
        B, T, C = x.size()
        self.device = x.device

        sr, k, v = self.jit_func(x, resolution)

        for j in range(self.recurrence):
            if j % 2 == 0:
                v = RUN_CUDA(self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v)
            else:
                h, w = resolution
                k = rearrange(k, "b (h w) c -> b (w h) c", h=h, w=w)
                v = rearrange(v, "b (h w) c -> b (w h) c", h=h, w=w)
                v = RUN_CUDA(self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v)
                k = rearrange(k, "b (w h) c -> b (h w) c", h=h, w=w)
                v = rearrange(v, "b (w h) c -> b (h w) c", h=h, w=w)

        x = v
        x = sr * x
        x = self.output(x)
        return x


class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, hidden_rate=4):
        super().__init__()
        self.n_embd = n_embd
        hidden_sz = int(hidden_rate * n_embd)
        self.key = nn.Linear(n_embd, hidden_sz, bias=False)

        self.omni_shift = OmniShift(dim=n_embd)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)

    def forward(self, x, resolution):

        h, w = resolution

        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)
        x = self.omni_shift(x)
        x = rearrange(x, "b c h w -> b (h w) c")

        k = self.key(x)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        x = torch.sigmoid(self.receptance(x)) * kv

        return x


class Block(nn.Module):
    def __init__(self, n_embd, hidden_rate=4):
        super().__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.att = VRWKV_SpatialMix(n_embd)

        self.ffn = VRWKV_ChannelMix(n_embd, hidden_rate)

        self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape

        resolution = (h, w)

        x = rearrange(x, "b c h w -> b (h w) c")
        x = x + self.gamma1 * self.att(self.ln1(x), resolution)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

        x = rearrange(x, "b c h w -> b (h w) c")
        x = x + self.gamma2 * self.ffn(self.ln2(x), resolution)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat, channel_scale=2):
        super(Downsample, self).__init__()
        mid_channels = n_feat * channel_scale // 4
        assert (
            mid_channels > 0
        ), "channel_scale must be such that n_feat * channel_scale is divisible by 4"
        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, mid_channels, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat, channel_scale=0.5):
        super(Upsample, self).__init__()
        mid_channels = int(n_feat * channel_scale * 4)
        self.body = nn.Sequential(
            nn.Conv2d(
                n_feat, mid_channels, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Restore_RWKV_Ref(nn.Module):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=8,
        loss_fun=nn.L1Loss(),
        scale=10,
    ):
        super().__init__()
        self.scale = scale
        self.loss_fun = loss_fun

        # ---- LR 上采样到 120×120 ----
        # 原 LR 为 48×48，目标 120×120，倍率 120/48 = 2.5
        self.lr_up = nn.Sequential(
            nn.Upsample(scale_factor=2.5, mode="bilinear", align_corners=False),
            nn.Conv2d(inp_channels, dim, 3, padding=1, bias=False),
        )

        # ---- 参考图像多尺度提取 (输出 120,60,30,15) ----
        # 先通过卷积从 480 降到 120 (两次 stride=2)
        self.ref_to_120 = nn.Sequential(
            nn.Conv2d(
                out_channels, dim, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
        # 然后连续使用 Downsample 得到 60,30,15
        self.ref_down_60 = Downsample(dim)  # 120 -> 60, 通道 dim -> 2*dim
        self.ref_down_30 = Downsample(dim * 2)  # 60  -> 30, 通道 2*dim -> 4*dim
        self.ref_down_15 = Downsample(dim * 4)  # 30  -> 15, 通道 4*dim -> 8*dim

        # ---- 各层级融合卷积 (1×1) ----
        self.fuse1 = nn.Conv2d(dim * 2, dim, 1)  # fea(120) + ref(120)
        self.fuse2 = nn.Conv2d(dim * 4, dim * 2, 1)  # e1下采样(60) + ref(60)
        self.fuse3 = nn.Conv2d(dim * 8, dim * 4, 1)  # e2下采样(30) + ref(30)
        self.fuse4 = nn.Conv2d(dim * 16, dim * 8, 1)  # e3下采样(15) + ref(15)

        # ---- 编码器 (分辨率从120开始, 下采样3次) ----
        self.encoder_level1 = nn.Sequential(
            *[Block(n_embd=dim) for _ in range(num_blocks[0])]
        )
        self.down1_2 = Downsample(dim)  # 120 → 60, 通道: dim → 2*dim
        self.encoder_level2 = nn.Sequential(
            *[Block(n_embd=dim * 2) for _ in range(num_blocks[1])]
        )
        self.down2_3 = Downsample(dim * 2)  # 60 → 30, 通道: 2*dim → 4*dim
        self.encoder_level3 = nn.Sequential(
            *[Block(n_embd=dim * 4) for _ in range(num_blocks[2])]
        )
        self.down3_4 = Downsample(dim * 4)  # 30 → 15, 通道: 4*dim → 8*dim
        self.latent = nn.Sequential(
            *[Block(n_embd=dim * 8) for _ in range(num_blocks[3])]
        )

        # ---- 解码器 (上采样3次，并融合跳跃连接) ----
        self.up4_3 = Upsample(dim * 8)  # 15 → 30, 通道: 8*dim → 4*dim
        self.reduce_chan_level3 = nn.Conv2d(
            4 * dim + 4 * dim, 4 * dim, 1
        )  # 拼接 up(4*dim) + e3(4*dim) → 8*dim → 4*dim
        self.decoder_level3 = nn.Sequential(
            *[Block(n_embd=dim * 4) for _ in range(num_blocks[2])]
        )

        self.up3_2 = Upsample(dim * 4)  # 30 → 60, 通道: 4*dim → 2*dim
        self.reduce_chan_level2 = nn.Conv2d(
            2 * dim + 2 * dim, 2 * dim, 1
        )  # 拼接 up(2*dim) + e2(2*dim) → 4*dim → 2*dim
        self.decoder_level2 = nn.Sequential(
            *[Block(n_embd=dim * 2) for _ in range(num_blocks[1])]
        )

        self.up2_1 = Upsample(dim * 2)  # 60 → 120, 通道: 2*dim → dim
        self.reduce_chan_level1 = nn.Conv2d(
            dim + dim, dim, 1
        )  # 拼接 up(dim) + e1(dim) → 2*dim → dim
        self.decoder_level1 = nn.Sequential(
            *[Block(n_embd=dim) for _ in range(num_blocks[0])]
        )

        self.refinement = nn.Sequential(
            *[Block(n_embd=dim) for _ in range(num_refinement_blocks)]
        )

        # ---- 最终上采样: 120 -> 480 (4倍) ----
        self.up_final = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 3, padding=1, bias=False),
            nn.PixelShuffle(2),  # dim -> dim, 分辨率*2
            nn.Conv2d(dim, dim * 4, 3, padding=1, bias=False),
            nn.PixelShuffle(2),  # dim -> dim, 分辨率*2  最终 120*4=480
        )
        # 高分辨率引导融合 (与原始HR残差)
        self.hr_refine = nn.Conv2d(
            out_channels * 2, out_channels, 3, padding=1
        )  # 无激活
        self.output_conv = nn.Conv2d(dim, out_channels, 3, padding=1, bias=True)

    def forward(self, lr1, hr1, lr2, label=None):
        # lr1, lr2: (B,3,48,48) , hr1: (B,3,480,480)
        B = lr1.shape[0]

        # 1. LR 上采样到 120×120
        fea = self.lr_up(lr1)  # (B,dim,120,120)

        # 2. 提取参考金字塔 (120,60,30,15)
        ref_120 = self.ref_to_120(hr1)  # (B,dim,120,120)
        ref_60 = self.ref_down_60(ref_120)  # (B,2*dim,60,60)
        ref_30 = self.ref_down_30(ref_60)  # (B,4*dim,30,30)
        ref_15 = self.ref_down_15(ref_30)  # (B,8*dim,15,15)

        # 3. 编码器 + 参考注入
        # 编码器
        e1 = self.encoder_level1(
            self.fuse1(torch.cat([fea, ref_120], dim=1))
        )  # (B,dim,120,120)
        e2 = self.encoder_level2(
            self.fuse2(torch.cat([self.down1_2(e1), ref_60], dim=1))
        )  # (B,2*dim,60,60)
        e3 = self.encoder_level3(
            self.fuse3(torch.cat([self.down2_3(e2), ref_30], dim=1))
        )  # (B,4*dim,30,30)
        l = self.latent(
            self.fuse4(torch.cat([self.down3_4(e3), ref_15], dim=1))
        )  # (B,8*dim,15,15)

        # 解码器
        d3 = self.decoder_level3(
            self.reduce_chan_level3(torch.cat([self.up4_3(l), e3], dim=1))
        )  # (B,4*dim,30,30)
        d2 = self.decoder_level2(
            self.reduce_chan_level2(torch.cat([self.up3_2(d3), e2], dim=1))
        )  # (B,2*dim,60,60)
        d1 = self.decoder_level1(
            self.reduce_chan_level1(torch.cat([self.up2_1(d2), e1], dim=1))
        )  # (B,dim,120,120)
        d1 = self.refinement(d1)

        # 5. 最终上采样到 480×480
        hr_feat = self.up_final(d1)  # (B,dim,480,480)
        out = self.output_conv(hr_feat)  # (B,3,480,480)

        # 6. 高分辨率参考引导 (直接使用原始HR)
        refine_input = torch.cat([out, hr1], dim=1)  # (B,6,480,480)
        residual = self.hr_refine(refine_input)  # (B,3,480,480)
        out = out + residual

        if label is None:
            return out
        else:
            return self.loss_fun(out, label)


# ---------- 测试（使用你的真实数据尺寸）----------
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Restore_RWKV_Ref(inp_channels=3, out_channels=3, dim=48, scale=10).to(
        device
    )

    # 模拟你的真实输入：LR 48×48, HR 参考 480×480
    lr1 = torch.randn((2, 3, 48, 48)).to(device)
    hr1 = torch.randn((2, 3, 480, 480)).to(device)
    lr2 = torch.randn((2, 3, 48, 48)).to(device)

    # 推理模式
    out = model(lr1, hr1, lr2)
    print(f"输出形状: {out.shape}")  # 应为 [2, 3, 480, 480]

    # 训练模式（需要 label）
    hr2 = torch.randn((2, 3, 480, 480)).to(device)
    loss = model(lr1, hr1, lr2, hr2)
    print(f"Loss: {loss.item()}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数: {total_params/1e6:.2f} M")


import pytorch_lightning as pl


class LitRestoreRWKV_Ref(pl.LightningModule):
    def __init__(
        self,
        inp_channels=3,
        out_channels=3,
        dim=48,
        num_blocks=[4, 6, 6, 8],
        num_refinement_blocks=8,
        scale=10,
        learning_rate=1e-4,
        warmup_steps=100,
        loss_fn=nn.L1Loss(),
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["loss_fn"])

        # 实例化原始模型
        self.model = Restore_RWKV_Ref(
            inp_channels=inp_channels,
            out_channels=out_channels,
            dim=dim,
            num_blocks=num_blocks,
            num_refinement_blocks=num_refinement_blocks,
            scale=scale,
            loss_fun=loss_fn,
        )

        self.criterion = loss_fn

    def forward(self, lr1, hr1, lr2, label=None):
        # 直接调用原始模型的 forward
        return self.model(lr1, hr1, lr2, label)

    def training_step(self, batch, batch_idx):
        lr1, hr1, lr2, hr2 = batch
        output = self(lr1, hr1, lr2)  # 输出 (B,3,H,W)
        loss = self.criterion(output, hr2)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        lr1, hr1, lr2, hr2 = batch
        output = self(lr1, hr1, lr2)
        loss = self.criterion(output, hr2)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        lr1, hr1, lr2, hr2 = batch
        output = self(lr1, hr1, lr2)
        loss = self.criterion(output, hr2)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return output, hr2  # 用于后续评估

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        # 正常使用 ReduceLROnPlateau，它会在每个 epoch 结束时根据 val_loss 调整学习率
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        # 在每次 optimizer step 之前，如果还在 warmup 阶段，手动调整学习率
        if self._step_count < self.warmup_steps:
            # 线性 warmup: 从 0 线性增长到 hparams.learning_rate
            lr_scale = min(1.0, (self._step_count + 1) / self.warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = self.hparams.learning_rate * lr_scale
        # 调用父类方法执行实际的 optimizer step
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)
        self._step_count += 1

    # ---------- 数据加载：需要在训练脚本中传入 DataLoader ----------
    def train_dataloader(self):
        return self._train_loader

    def val_dataloader(self):
        return self._val_loader

    def test_dataloader(self):
        return self._test_loader

    def set_dataloaders(self, train_loader, val_loader, test_loader):
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._test_loader = test_loader
