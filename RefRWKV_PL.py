"""分阶段训练的 Lightning 模块：LitRefSRWKV / LitRefDiffRWKV / LitEnRWKV"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW


# ============================================================
# 1. RefSRWKV — 超分模块（保守回归训练）
# ============================================================
class LitRefSRWKV(pl.LightningModule):
    def __init__(
        self,
        model_sr: nn.Module,
        learning_rate: float = 1e-4,
        warmup_steps: int = 100,
        loss_fn: nn.Module = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model_sr", "loss_fn"])

        self.model_sr = model_sr
        self.criterion = loss_fn or nn.L1Loss()
        self._step_count = 0

    def forward(self, lr, ref):
        return self.model_sr(lr, ref)

    def training_step(self, batch, batch_idx):
        lr, hr, ref = batch
        output = self(lr, ref)
        loss = self.criterion(output, hr)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        lr, hr, ref = batch
        output = self(lr, ref)
        loss = self.criterion(output, hr)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        lr, hr, ref = batch
        output = self(lr, ref)
        loss = self.criterion(output, hr)
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        return output, hr

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
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
        # Step 级 warmup
        if self._step_count < self.hparams.warmup_steps:
            lr_scale = min(1.0, (self._step_count + 1) / self.hparams.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = self.hparams.learning_rate * lr_scale
        if optimizer_closure is not None:
            optimizer.step(closure=optimizer_closure)
        else:
            optimizer.step()
        self._step_count += 1

    def on_train_start(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"✅ LitRefSRWKV 训练开始 | 参数量: {total / 1e6:.2f}M")


# ============================================================
# 2. RefDiffRWKV — 扩散模块（SNR 加权 + CFG + Cosine 退火）
# ============================================================
class LitRefDiffRWKV(pl.LightningModule):
    def __init__(
        self,
        model_diff: nn.Module,
        num_timesteps: int = 1000,
        cfg_drop_prob: float = 0.1,
        learning_rate: float = 2e-4,
        weight_decay: float = 1e-2,
        beta1: float = 0.9,
        beta2: float = 0.999,
        warmup_epochs: int = 5,
        scheduler: str = "cosine",
        eta_min: float = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model_diff"])

        self.model_diff = model_diff
        self.num_timesteps = num_timesteps
        self.cfg_drop_prob = cfg_drop_prob
        self.lr = learning_rate
        self.wd = weight_decay
        self.b1, self.b2 = beta1, beta2
        self.warmup_epochs = warmup_epochs
        self.scheduler_type = scheduler
        self.eta_min = eta_min if eta_min is not None else learning_rate * 0.01

    def _add_noise(self, x0, noise, t):
        s = 0.008
        T = self.num_timesteps
        alpha_bar = torch.cos(((t.float() / T + s) / (1 + s)) * math.pi / 2) ** 2
        alpha_bar = alpha_bar.view(-1, 1, 1, 1)
        x_t = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise
        return x_t, noise, alpha_bar

    def forward(self, x_t, timesteps, LR, Ref):
        return self.model_diff(x_t, timesteps, LR, Ref)

    def _step(self, batch, stage):
        lr, hr, ref = batch
        B = hr.shape[0]
        device = hr.device

        t = torch.randint(1, self.num_timesteps + 1, (B,), device=device)
        noise = torch.randn_like(hr)
        x_t, noise, alpha_bar = self._add_noise(hr, noise, t)

        # CFG: 训练时逐样本随机丢弃 ref
        if self.training:
            drop_mask = torch.rand(B, device=device) < self.cfg_drop_prob
            drop_mask_exp = drop_mask.view(B, 1, 1, 1)
            ref_cond = torch.where(drop_mask_exp, torch.zeros_like(ref), ref)
        else:
            ref_cond = ref

        pred_noise = self.model_diff(x_t, t, lr, ref_cond)

        # SNR 加权 MSE，gamma 截断
        snr = alpha_bar / (1 - alpha_bar + 1e-8)
        gamma = 5.0
        loss_weight = torch.minimum(snr, torch.full_like(snr, gamma))
        loss = (loss_weight * ((pred_noise - noise) ** 2)).mean()

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
            betas=(self.b1, self.b2),
            weight_decay=self.wd,
            eps=1e-8,
        )
        total_epochs = self.trainer.max_epochs if self.trainer else 200

        def lr_lambda(epoch):
            if epoch < self.warmup_epochs:
                return float(epoch + 1) / float(max(1, self.warmup_epochs))
            progress = float(epoch - self.warmup_epochs) / float(
                max(1, total_epochs - self.warmup_epochs)
            )
            if self.scheduler_type == "cosine":
                ratio = self.eta_min / self.lr + (1.0 - self.eta_min / self.lr) * (
                    0.5 * (1.0 + math.cos(math.pi * progress))
                )
            else:
                ratio = 1.0 - (1.0 - self.eta_min / self.lr) * progress
            return max(0.0, ratio)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch", "frequency": 1},
        }

    def on_train_start(self):
        total = sum(p.numel() for p in self.parameters())
        print(f"✅ LitRefDiffRWKV 训练开始 | 参数量: {total / 1e6:.2f}M")


# ============================================================
# 3. EnRWKV — 增强模块（依赖冻结扩散模型，t 范围限制）
# ============================================================
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

