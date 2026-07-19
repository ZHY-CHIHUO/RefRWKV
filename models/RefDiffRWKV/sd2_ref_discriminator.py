"""
sd2_ref_discriminator.py — 双判别器：语义 D + 纹理一致性 D
与 SD2RefGenerator 分离，独立 LightningModule
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from typing import Optional, List, Tuple

import open_clip
from vision_aided_loss.cv_discriminator import BlurPool, spectral_norm
from vision_aided_loss.cv_losses import multilevel_loss

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
#  Haar 小波高频分解
# ══════════════════════════════════════════════════════════════


def haar_highpass(x: torch.Tensor) -> torch.Tensor:
    B, C, H, W = x.shape
    assert C == 3, f"haar_highpass 需要 3 通道输入，收到 {C} 通道"
    if H % 2 != 0:
        x = x[:, :, : H - 1, :]
        H = H - 1
    if W % 2 != 0:
        x = x[:, :, :, : W - 1]
        W = W - 1
    x = x.reshape(B, C, H // 2, 2, W // 2, 2)
    x00 = x[:, :, :, 0, :, 0]
    x01 = x[:, :, :, 0, :, 1]
    x10 = x[:, :, :, 1, :, 0]
    x11 = x[:, :, :, 1, :, 1]
    high = torch.cat([x01 - x00, x10 - x00, x11 - x00], dim=1)
    std = high.std(dim=[2, 3], keepdim=True).clamp(min=0.1)
    return ((high - high.mean(dim=[2, 3], keepdim=True)) / std).clamp(-10, 10)


# ══════════════════════════════════════════════════════════════
#  Backbone: ConvNeXt-Base-W
# ══════════════════════════════════════════════════════════════


def _visual_forward(model, image, return_feats=False, return_pooled_feats=False):
    x, intermediates = model.trunk.forward_intermediates(
        image,
        indices=None,
        norm=False,
        stop_early=False,
        intermediates_only=False,
    )
    if return_feats:
        return intermediates[1:]
    x = model.trunk.forward_head(x)
    x = model.head(x)
    if return_pooled_feats:
        intermediates[-1] = x
        return intermediates[1:]
    return x


class ImageOpenCLIPConvNext(nn.Module):
    def __init__(self, precision="fp32", trainable_stages=0):
        super().__init__()
        full_model, _, _ = open_clip.create_model_and_transforms(
            "convnext_base_w",
            pretrained="laion2b_s13b_b82k",
            precision=precision,
        )
        self.model = full_model.visual
        self.trainable_stages = trainable_stages
        self.model.eval().requires_grad_(False)
        if trainable_stages >= 1:
            self.model.trunk.stem.requires_grad_(True)
        if trainable_stages >= 2:
            self.model.trunk.stages[0].requires_grad_(True)

    def encode_image(self, image, return_feats=False, return_pooled_feats=False):
        return _visual_forward(self.model, image, return_feats, return_pooled_feats)


# ══════════════════════════════════════════════════════════════
#  MultiLevelDConv
# ══════════════════════════════════════════════════════════════


class MultiLevelDConv(nn.Module):
    def __init__(
        self,
        level=3,
        in_ch1=(256, 512),
        in_ch2=640,
        out_ch=256,
        num_classes=0,
        activation=nn.LeakyReLU(0.2, inplace=False),  # inplace=False 避免兼容性问题
        down=1,
    ):
        super().__init__()
        self.decoder = nn.ModuleList()
        self.level = level
        self.in_ch1 = in_ch1

        for i in range(level - 1):
            layers = [
                (
                    BlurPool(in_ch1[i], pad_type="zero", stride=1, pad_off=1)
                    if down > 1
                    else nn.Identity()
                ),
                spectral_norm(
                    nn.Conv2d(
                        in_ch1[i],
                        out_ch,
                        3,
                        stride=2 if down > 1 else 1,
                        padding=1 if down == 1 else 0,
                    )
                ),
                activation,
                spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1)),
                activation,
                BlurPool(out_ch, pad_type="zero", stride=1),
                spectral_norm(nn.Conv2d(out_ch, 1, kernel_size=1, stride=2)),
                nn.Tanh(),
            ]
            self.decoder.append(nn.Sequential(*layers))
        self.decoder.append(
            nn.Sequential(spectral_norm(nn.Linear(in_ch2, out_ch)), activation)
        )
        self.out = spectral_norm(nn.Linear(out_ch, 1))
        self.embed = nn.Embedding(num_classes, out_ch) if num_classes > 0 else None

    def forward(self, x, c=None):
        final_pred = []
        for i in range(self.level - 1):
            assert (
                x[i].shape[1] == self.in_ch1[i]
            ), f"Channel mismatch at level {i}: expected {self.in_ch1[i]}, got {x[i].shape[1]}"
            final_pred.append(self.decoder[i](x[i]).squeeze(1))
        h = self.decoder[-1](x[-1].float())
        out = self.out(h)
        if self.embed is not None and c is not None:
            out += torch.sum(self.embed(c) * h, dim=1, keepdim=True)
        final_pred.append(torch.tanh(out))
        return final_pred


# ══════════════════════════════════════════════════════════════
#  ImageConvNextDiscriminator（语义判别器）
# ══════════════════════════════════════════════════════════════


class ImageConvNextDiscriminator(nn.Module):
    def __init__(self, alpha=0.8, precision="fp32", use_freq=True, trainable_stages=1):
        super().__init__()
        self.gan_alpha = alpha
        self.use_freq = use_freq
        self.trainable_stages = trainable_stages

        self.model = ImageOpenCLIPConvNext(
            precision=precision, trainable_stages=trainable_stages
        )
        self.decoder = MultiLevelDConv(
            level=3, in_ch1=[256, 512], in_ch2=640, out_ch=256, down=2
        )

        if use_freq:
            self._adapt_first_conv(9)

        self.register_buffer(
            "image_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32),
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32),
        )

    def _adapt_first_conv(self, in_channels=9):
        old_conv = self.model.model.trunk.stem[0]
        old_weight = old_conv.weight.data.clone()
        new_conv = nn.Conv2d(
            in_channels, old_weight.shape[0], kernel_size=4, stride=4, padding=0
        )
        with torch.no_grad():
            new_conv.weight[:, :3] = old_weight[:, :3]
            for c in range(3, in_channels):
                nn.init.orthogonal_(new_conv.weight[:, c : c + 1])
        new_conv.bias.data.copy_(old_conv.bias.data)
        new_conv.requires_grad_(self.trainable_stages >= 1)
        self.model.model.trunk.stem[0] = new_conv
        if self.trainable_stages >= 1:
            self.model.model.trunk.stem[1].requires_grad_(True)

    def train(self, mode=True):
        super().train(mode)  # 调用父类
        self.decoder.train(mode)
        if self.trainable_stages >= 1:
            self.model.model.trunk.stem.train(mode)
        if self.trainable_stages >= 2:
            self.model.model.trunk.stages[0].train(mode)
        return self

    def eval(self):
        return self.train(False)

    def requires_grad_(self, requires_grad=True):
        self.decoder.requires_grad_(requires_grad)
        if self.trainable_stages >= 1:
            self.model.model.trunk.stem.requires_grad_(requires_grad)
        if self.trainable_stages >= 2:
            self.model.model.trunk.stages[0].requires_grad_(requires_grad)
        return self

    def forward(self, x, for_real=True, for_G=False, return_logits=False):
        assert x.shape[1] == 3, f"语义 D 期望 3 通道输入，收到 {x.shape[1]}"
        if torch.isnan(x).any() or torch.isinf(x).any():
            raise ValueError(f"语义 D 输入包含 NaN/Inf: min={x.min()}, max={x.max()}")

        if self.use_freq:
            x = haar_highpass(x)
        else:
            x = x * 0.5 + 0.5
            x = (x - self.image_mean[:, None, None]) / self.image_std[:, None, None]

        try:
            features = self.model.encode_image(x, return_pooled_feats=True)
            features = self.decoder(features)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("语义 D CUDA OOM，尝试清理缓存")
                torch.cuda.empty_cache()
                zero = torch.zeros((), device=x.device, dtype=x.dtype)
                return zero if not return_logits else (zero, [])
            raise

        loss_fn = multilevel_loss(alpha=self.gan_alpha)
        loss = loss_fn(features, for_real=for_real, for_G=for_G)
        return (loss, features) if return_logits else loss


# ══════════════════════════════════════════════════════════════
#  TextureConsistencyDiscriminator（纹理一致性判别器）
# ══════════════════════════════════════════════════════════════


class TextureConsistencyDiscriminator(nn.Module):
    def __init__(self, in_ch=3, base_ch=48, num_scales=4, use_spectral=True):
        super().__init__()
        self.num_scales = num_scales
        self._sn = spectral_norm if use_spectral else lambda x: x

        enc_chs = [in_ch] + [base_ch * (2**i) for i in range(num_scales)]
        self.encoder = nn.ModuleList()
        for i in range(num_scales):
            ch_in, ch_out = enc_chs[i], enc_chs[i + 1]
            self.encoder.append(
                nn.Sequential(
                    self._sn(nn.Conv2d(ch_in, ch_out, 4, 2, 1)),
                    nn.GroupNorm(min(8, ch_out // 4), ch_out),
                    nn.LeakyReLU(0.2, inplace=False),
                    self._sn(nn.Conv2d(ch_out, ch_out, 3, 1, 1)),
                    nn.GroupNorm(min(8, ch_out // 4), ch_out),
                    nn.LeakyReLU(0.2, inplace=False),
                )
            )

        self.scale_heads = nn.ModuleList()
        for i in range(num_scales):
            ch = enc_chs[i + 1]
            self.scale_heads.append(
                nn.Sequential(
                    self._sn(nn.Conv2d(ch, ch // 2, 3, 1, 1)),
                    nn.LeakyReLU(0.2, inplace=False),
                    self._sn(nn.Conv2d(ch // 2, ch // 4, 3, 1, 1)),
                    nn.LeakyReLU(0.2, inplace=False),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    self._sn(nn.Linear(ch // 4, 1)),
                )
            )
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)

    def _extract_features(self, x):
        feats = []
        for layer in self.encoder:
            x = layer(x)
            feats.append(x)
        return feats

    def forward(self, image, ref):
        if torch.isnan(image).any() or torch.isinf(image).any():
            raise ValueError("纹理 D image 输入包含 NaN/Inf")
        if torch.isnan(ref).any() or torch.isinf(ref).any():
            raise ValueError("纹理 D ref 输入包含 NaN/Inf")

        if image.shape[-2:] != ref.shape[-2:]:
            ref = F.interpolate(
                ref, size=image.shape[-2:], mode="bilinear", align_corners=False
            )

        feats_image = self._extract_features(image)
        feats_ref = self._extract_features(ref)

        per_scale_logits = []
        for i, head in enumerate(self.scale_heads):
            per_scale_logits.append(head(torch.abs(feats_image[i] - feats_ref[i])))

        weights = torch.softmax(self.scale_weights, dim=0)
        logits_stacked = torch.stack(per_scale_logits, dim=0)
        weight_shape = [-1] + [1] * (logits_stacked.ndim - 1)
        weighted = (weights.view(*weight_shape) * logits_stacked).sum(dim=0)

        return torch.clamp(weighted, -5.0, 5.0), [l.detach() for l in per_scale_logits]


# ══════════════════════════════════════════════════════════════
#  SD2RefDiscriminator — LightningModule 封装
# ══════════════════════════════════════════════════════════════


class SD2RefDiscriminator(LightningModule):
    """双判别器封装：语义 D + 纹理 D"""

    def __init__(
        self,
        use_semantic_d: bool = True,
        use_texture_d: bool = True,
        semantic_alpha: float = 0.8,
        semantic_use_freq: bool = True,
        semantic_trainable_stages: int = 1,
        semantic_precision: str = "fp32",
        texture_base_ch: int = 48,
        texture_num_scales: int = 4,
        texture_use_spectral: bool = True,
        lr_semantic: float = 5e-6,
        lr_texture: float = 1e-6,
        weight_decay: float = 1e-3,
        betas: Tuple[float, float] = (0.5, 0.999),
    ):
        super().__init__()
        self.save_hyperparameters()
        self.use_semantic_d = use_semantic_d
        self.use_texture_d = use_texture_d

        self.D_sem = (
            ImageConvNextDiscriminator(
                alpha=semantic_alpha,
                precision=semantic_precision,
                use_freq=semantic_use_freq,
                trainable_stages=semantic_trainable_stages,
            )
            if use_semantic_d
            else None
        )

        self.D_tex = (
            TextureConsistencyDiscriminator(
                in_ch=3,
                base_ch=texture_base_ch,
                num_scales=texture_num_scales,
                use_spectral=texture_use_spectral,
            )
            if use_texture_d
            else None
        )

        self._d_sem_accum_count = 0
        self._d_tex_accum_count = 0

    def train(self, mode: bool = True):
        super().train(mode)  # [FIX] 调用父类
        if self.D_sem is not None:
            self.D_sem.train(mode)
        if self.D_tex is not None:
            self.D_tex.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def requires_grad_(self, requires_grad: bool = True):
        if self.D_sem is not None:
            self.D_sem.requires_grad_(requires_grad)
        if self.D_tex is not None:
            self.D_tex.requires_grad_(requires_grad)
        return self

    def _zero_loss(self, ref_tensor: torch.Tensor) -> torch.Tensor:
        return torch.zeros((), device=ref_tensor.device, dtype=ref_tensor.dtype)

    def _validate_inputs(self, *tensors):
        for i, t in enumerate(tensors):
            if t is None:
                continue
            if torch.isnan(t).any() or torch.isinf(t).any():
                raise ValueError(
                    f"Input tensor {i} contains NaN/Inf: min={t.min().item()}, max={t.max().item()}"
                )

    def compute_g_loss(self, fake, ref=None, lambda_semantic=1.0, lambda_texture=1.0):
        self._validate_inputs(fake, ref)
        loss = self._zero_loss(fake)
        with torch.amp.autocast("cuda", enabled=False):
            if self.use_semantic_d and lambda_semantic > 0:
                loss = (
                    loss
                    + lambda_semantic
                    * self.D_sem(fake.float(), for_real=False, for_G=True).mean()
                )
            if self.use_texture_d and lambda_texture > 0 and ref is not None:
                ref_d = ref.detach().float()
                fake_logit, _ = self.D_tex(fake.float(), ref_d)
                loss = loss + lambda_texture * (-fake_logit.mean())
        return loss

    def compute_d_loss(
        self, real, fake, ref=None, lambda_semantic=1.0, lambda_texture=1.0
    ):
        self._validate_inputs(real, fake, ref)
        real, fake = real.detach(), fake.detach()
        ref = ref.detach() if ref is not None else None
        loss = self._zero_loss(real)
        with torch.amp.autocast("cuda", enabled=False):
            if self.use_semantic_d and lambda_semantic > 0:
                r, f = real.float(), fake.float()
                loss = loss + lambda_semantic * (
                    self.D_sem(r, for_real=True).mean()
                    + self.D_sem(f, for_real=False).mean()
                )
            if self.use_texture_d and lambda_texture > 0 and ref is not None:
                ref_f = ref.float()  # 缓存避免重复转换
                real_logit, _ = self.D_tex(real.float(), ref_f)
                fake_logit, _ = self.D_tex(fake.float(), ref_f)
                loss = loss + lambda_texture * (
                    F.relu(1.0 - real_logit).mean() + F.relu(1.0 + fake_logit).mean()
                )
        return loss

    def forward(self, real, fake, ref=None, mode="both"):
        log_dict = {}
        loss_D = self._zero_loss(real)
        if mode in ("both", "semantic") and self.use_semantic_d:
            l = self.compute_d_loss(
                real, fake, ref=None, lambda_semantic=1.0, lambda_texture=0.0
            )
            loss_D = loss_D + l
            log_dict["loss_D_sem"] = l.detach()
        if mode in ("both", "texture") and self.use_texture_d and ref is not None:
            l = self.compute_d_loss(
                real, fake, ref=ref, lambda_semantic=0.0, lambda_texture=1.0
            )
            loss_D = loss_D + l
            log_dict["loss_D_tex"] = l.detach()
        log_dict["loss_D_total"] = loss_D.detach()
        return loss_D, log_dict

    def training_step(self, batch, batch_idx):
        try:
            real, fake = batch["hr"], batch["sr"].detach()
            ref = batch.get("ref", None)
            loss, log_dict = self.forward(real, fake, ref=ref, mode="both")
            self.log_dict(
                {f"train/{k}": v for k, v in log_dict.items()},
                on_step=True,
                prog_bar=True,
            )
            return loss
        except (ValueError, RuntimeError) as e:
            logger.warning("D training_step 异常: %s", e)
            torch.cuda.empty_cache()
            return self._zero_loss(batch["hr"])

    def configure_optimizers(self):
        opts = []
        if self.use_semantic_d:
            ps = [p for p in self.D_sem.parameters() if p.requires_grad]
            if ps:
                opts.append(
                    torch.optim.AdamW(
                        ps,
                        lr=self.hparams.lr_semantic,
                        betas=self.hparams.betas,
                        weight_decay=self.hparams.weight_decay,
                    )
                )
            else:
                logger.warning("D_sem 无可训练参数，跳过 D_sem 优化器")
        if self.use_texture_d:
            ps = list(self.D_tex.parameters())
            if ps:
                opts.append(
                    torch.optim.AdamW(
                        ps,
                        lr=self.hparams.lr_texture,
                        betas=self.hparams.betas,
                        weight_decay=self.hparams.weight_decay,
                    )
                )
            else:
                logger.warning("D_tex 无可训练参数，跳过 D_tex 优化器")
        return opts

    def on_save_checkpoint(self, checkpoint):
        checkpoint["d_sem_accum_count"] = self._d_sem_accum_count
        checkpoint["d_tex_accum_count"] = self._d_tex_accum_count

    def on_load_checkpoint(self, checkpoint):
        self._d_sem_accum_count = checkpoint.get("d_sem_accum_count", 0)
        self._d_tex_accum_count = checkpoint.get("d_tex_accum_count", 0)


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SD2RefDiscriminator(use_semantic_d=True, use_texture_d=True).to(device)
    model.eval()
    B, B3 = 2, 3
    real = torch.randn(B, B3, 480, 480, device=device)
    fake = torch.randn(B, B3, 480, 480, device=device)
    ref = torch.randn(B, B3, 480, 480, device=device)
    print("D loss:", model.compute_d_loss(real, fake, ref=ref).item())
    print("G loss:", model.compute_g_loss(fake, ref=ref).item())
    loss, logs = model(real, fake, ref=ref, mode="both")
    print("forward loss:", loss.item())
    print("logs:", {k: v.item() for k, v in logs.items()})
    loss.backward()
    print("backward ok")
