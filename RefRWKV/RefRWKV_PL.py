import sys
from pathlib import Path

# 添加项目根目录到 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from RefSRWKV.models import RefSRWKV, RefDiffRWKV, EnRWKV

import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR


class RefDiffRWKV_PL(pl.LightningModule):

    def __init__(
        self,
        model,  # RefDiffRWKV 实例
        lr: float = 4e-4,
        weight_decay: float = 1e-2,
        beta1: float = 0.9,
        beta2: float = 0.999,
        warmup_epochs: int = 5,
        scheduler: str = "cosine",  # "cosine" 或 "linear"
        num_timesteps: int = 1000,
        eta_min: float = None,  # 最小学习率因子，默认 lr * 0.01
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_epochs = warmup_epochs
        self.scheduler_type = scheduler
        self.num_timesteps = num_timesteps
        self.eta_min = eta_min if eta_min is not None else lr * 0.01

    def forward(self, x_t, timesteps, LR, Ref):
        return self.model(x_t, timesteps, LR, Ref)

    def _add_noise(self, x0, noise, t):
        s = 0.008
        T = self.num_timesteps
        alpha_bar = torch.cos(((t.float() / T + s) / (1 + s)) * math.pi / 2) ** 2
        alpha_bar = alpha_bar.view(-1, 1, 1, 1)
        x_t = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise
        return x_t, noise, alpha_bar   # 返回 alpha_bar

    def _compute_loss(self, batch, stage: str = "train"):
        lr, hr, ref = batch
        B = hr.shape[0]

        if stage == "train":
            p_replace = 0.1
            mask_replace = torch.rand(B, device=self.device) < p_replace
            ref = torch.where(mask_replace.view(-1, 1, 1, 1), hr, ref)
        else:
            mask_replace = torch.zeros(B, dtype=torch.bool, device=self.device)

        t = torch.randint(1, self.num_timesteps, (B,), device=self.device)
        noise = torch.randn_like(hr)
        x_t, noise, alpha_bar = self._add_noise(hr, noise, t)

        pred_noise = self.model(x_t, t, lr, ref)
        loss_per_sample = ((pred_noise - noise) ** 2).mean(dim=[1, 2, 3])

        if stage == "train":
            alpha_bar_flat = alpha_bar.view(B)
            snr = alpha_bar_flat / (1 - alpha_bar_flat + 1e-8)
            weight = snr ** 0.5
            weight = weight / weight.mean()
            weight = weight.clone()
            weight[mask_replace] = weight[mask_replace] * 2.0
        else:
            weight = torch.ones(B, device=self.device)

        loss = (loss_per_sample * weight).mean()

        self.log(f"{stage}-loss", loss, prog_bar=True, sync_dist=True,
                on_step=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._compute_loss(batch, "train")

        if self.trainer.optimizers:
            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("lr", current_lr, prog_bar=True, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._compute_loss(batch, "val")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._compute_loss(batch, "test")
        return loss

    def on_train_epoch_end(self):
        if self.trainer.global_rank == 0:
            train_loss = self.trainer.callback_metrics.get("train-loss_epoch", 0.0)
            print(f"Epoch {self.current_epoch:04d} | Train Loss: {train_loss:.6f}")

    def on_validation_epoch_end(self):
        if self.trainer.global_rank == 0:
            val_loss = self.trainer.callback_metrics.get("val-loss", 0.0)
            current_lr = (
                self.trainer.optimizers[0].param_groups[0]["lr"]
                if self.trainer.optimizers
                else 0.0
            )
            print(
                f"Epoch {self.current_epoch:04d} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e}"
            )

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            betas=(self.hparams.beta1, self.hparams.beta2),
            weight_decay=self.weight_decay,
            eps=1e-8,
        )

        # 从 Trainer 获取总 epoch 数，若不可用则回退到默认值
        total_epochs = self.trainer.max_epochs if self.trainer else 200
        warmup_epochs = self.warmup_epochs
        eta_min = self.eta_min
        scheduler_type = self.scheduler_type
        init_lr = self.lr

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            progress = float(epoch - warmup_epochs) / float(
                max(1, total_epochs - warmup_epochs)
            )
            if scheduler_type == "cosine":
                # 余弦退火，终点为 eta_min/init_lr
                lr_ratio = eta_min / init_lr + (1.0 - eta_min / init_lr) * (
                    0.5 * (1.0 + math.cos(math.pi * progress))
                )
            else:
                # 线性衰减
                lr_ratio = 1.0 - (1.0 - eta_min / init_lr) * progress
            return max(0.0, lr_ratio)

        scheduler = LambdaLR(optimizer, lr_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_train_start(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        total_epochs = self.trainer.max_epochs if self.trainer else "?"
        print(f"✅ RefDiffRWKV_PL Training Started!")
        print(f"   Total Parameters: {total_params / 1e6:.2f}M")
        print(f"   Learning Rate: {self.lr}")
        print(f"   Warmup Epochs: {self.warmup_epochs}")
        print(f"   Total Epochs: {total_epochs}")
        print(f"   Scheduler Type: {self.scheduler_type}")