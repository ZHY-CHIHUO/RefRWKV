import math
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

class RefRWKV_PL(pl.LightningModule):
    """
    集成 RefSRWKV (Better Start) + RefDiffRWKV (主扩散) + EnRWKV (增强) 的 Lightning 模块。
    支持分阶段训练，也支持同步联合训练。
    """

    def __init__(
        self,
        model_sr: nn.Module,
        model_diff: nn.Module,
        model_enhance: nn.Module,
        # 训练开关
        train_sr: bool = True,
        train_diff: bool = True,
        train_enhance: bool = True,
        # 扩散参数
        num_timesteps: int = 1000,
        # 学习率（可单独指定）
        lr_sr: float = 1e-4,
        lr_diff: float = 4e-4,
        lr_enhance: float = 1e-4,
        weight_decay: float = 1e-2,
        beta1: float = 0.9,
        beta2: float = 0.999,
        warmup_epochs: int = 5,
        scheduler: str = "cosine",
        eta_min: float = None,
        # 损失权重
        loss_sr_weight: float = 0.1,        # 超分损失权重
        loss_enhance_weight: float = 0.1,   # 增强损失权重
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["model_sr", "model_diff", "model_enhance"])

        self.model_sr = model_sr
        self.model_diff = model_diff
        self.model_enhance = model_enhance

        self.train_sr = train_sr
        self.train_diff = train_diff
        self.train_enhance = train_enhance
        self.num_timesteps = num_timesteps
        self.loss_sr_weight = loss_sr_weight
        self.loss_enhance_weight = loss_enhance_weight

        # 设置各模块是否可训练
        self._set_requires_grad()

        # 学习率参数
        self.lr_sr = lr_sr
        self.lr_diff = lr_diff
        self.lr_enhance = lr_enhance
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.warmup_epochs = warmup_epochs
        self.scheduler_type = scheduler
        self.eta_min = eta_min if eta_min is not None else lr_diff * 0.01

    def _set_requires_grad(self):
        """根据训练开关冻结/解冻各模块参数"""
        for param in self.model_sr.parameters():
            param.requires_grad = self.train_sr
        for param in self.model_diff.parameters():
            param.requires_grad = self.train_diff
        for param in self.model_enhance.parameters():
            param.requires_grad = self.train_enhance

    def _add_noise(self, x0, noise, t):
        """添加噪声的余弦调度 (同原代码)"""
        s = 0.008
        T = self.num_timesteps
        alpha_bar = torch.cos(((t.float() / T + s) / (1 + s)) * math.pi / 2) ** 2
        alpha_bar = alpha_bar.view(-1, 1, 1, 1)
        x_t = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise
        return x_t, noise, alpha_bar

    def forward(self, x_t, timesteps, LR, Ref):
        """
        推理时使用（训练时使用 _compute_loss 直接计算所有损失）。
        若需要采样，请单独调用外部采样函数。
        """
        return self.model_diff(x_t, timesteps, LR, Ref)

    def _compute_loss(self, batch, stage="train"):
        lr, hr, ref = batch
        B = hr.shape[0]
        device = hr.device

        # ------------------ 1. 超分损失 (RefSRWKV) ------------------
        loss_sr = 0.0
        if self.train_sr:
            # 使用 LR, Ref 生成初始超分图像 I_start
            I_start = self.model_sr(lr, ref, label=None)  # 假设 forward 接受 (lr, ref)
            loss_sr = F.l1_loss(I_start, hr)  # 可与 HR 直接计算 L1 损失

        # ------------------ 2. 扩散损失 (RefDiffRWKV) ------------------
        t = torch.randint(1, self.num_timesteps, (B,), device=device)
        noise = torch.randn_like(hr)
        x_t, noise, alpha_bar = self._add_noise(hr, noise, t)
        pred_noise = self.model_diff(x_t, t, lr, ref)
        loss_diff = ((pred_noise - noise) ** 2).mean()

        # ------------------ 3. 增强损失 (EnRWKV) ------------------
        loss_enhance = 0.0
        if self.train_enhance:
            # 使用扩散模型预测的噪声计算一步去噪后的图像 pred_x0
            with torch.no_grad():
                alpha_bar_t = alpha_bar.view(-1, 1, 1, 1)
                pred_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * pred_noise) / torch.sqrt(alpha_bar_t)
                pred_x0 = torch.clamp(pred_x0, -1, 1)  # 确保在合法范围

            # 将 pred_x0 作为增强模型的输入（也可同时传入 ref 或其它条件）
            refined = self.model_enhance(pred_x0, label=None)  # 输入为单图像
            loss_enhance = F.l1_loss(refined, hr)

        # ------------------ 4. 总损失 ------------------
        total_loss = loss_diff + self.loss_sr_weight * loss_sr + self.loss_enhance_weight * loss_enhance

        # 日志记录
        self.log(f"{stage}-loss_sr", loss_sr, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{stage}-loss_diff", loss_diff, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{stage}-loss_enhance", loss_enhance, prog_bar=True, on_step=True, on_epoch=True)
        self.log(f"{stage}-loss_total", total_loss, prog_bar=True, on_step=True, on_epoch=True)

        return total_loss

    def training_step(self, batch, batch_idx):
        return self._compute_loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._compute_loss(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._compute_loss(batch, "test")

    def configure_optimizers(self):
        # 为不同模块设置不同学习率
        param_groups = []
        if self.train_sr:
            param_groups.append({'params': self.model_sr.parameters(), 'lr': self.lr_sr})
        if self.train_diff:
            param_groups.append({'params': self.model_diff.parameters(), 'lr': self.lr_diff})
        if self.train_enhance:
            param_groups.append({'params': self.model_enhance.parameters(), 'lr': self.lr_enhance})

        optimizer = AdamW(
            param_groups,
            lr=self.lr_diff,  # 默认值，会被实际 lr 覆盖
            betas=(self.beta1, self.beta2),
            weight_decay=self.weight_decay,
            eps=1e-8,
        )

        # 从 Trainer 获取总 epoch 数
        total_epochs = self.trainer.max_epochs if self.trainer else 200
        warmup_epochs = self.warmup_epochs
        eta_min = self.eta_min
        scheduler_type = self.scheduler_type
        init_lr = self.lr_diff  # 仅用于计算比例

        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(max(1, warmup_epochs))
            progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            if scheduler_type == "cosine":
                lr_ratio = eta_min / init_lr + (1.0 - eta_min / init_lr) * (
                    0.5 * (1.0 + math.cos(math.pi * progress))
                )
            else:
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
        total_params = sum(p.numel() for p in self.parameters())
        print(f"✅ RefRWKV_PL Training Started!")
        print(f"   Total Parameters: {total_params / 1e6:.2f}M")
        print(f"   Train SR: {self.train_sr}")
        print(f"   Train Diff: {self.train_diff}")
        print(f"   Train Enhance: {self.train_enhance}")