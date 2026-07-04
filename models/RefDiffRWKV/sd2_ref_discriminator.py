"""
sd2_ref_discriminator.py — 双判别器：语义 D + 纹理一致性 D
与 SD2RefGenerator 分离，独立 LightningModule
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from typing import Optional, List, Tuple

import open_clip
from vision_aided_loss.cv_discriminator import BlurPool, spectral_norm
from vision_aided_loss.cv_losses import multilevel_loss


# ══════════════════════════════════════════════════════════════
#  Haar 小波高频分解
# ══════════════════════════════════════════════════════════════
def haar_highpass(x: torch.Tensor) -> torch.Tensor:
    """Haar 小波高频分解"""
    B, C, H, W = x.shape

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

    LH = x01 - x00
    HL = x10 - x00
    HH = x11 - x00

    high = torch.cat([LH, HL, HH], dim=1)
    return high


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

        model_name = "convnext_base_w"
        pretrained_name = "laion2b_s13b_b82k"

        full_model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained_name,
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
        activation=nn.LeakyReLU(0.2, inplace=True),
        down=1,
    ):
        super().__init__()
        self.decoder = nn.ModuleList()
        self.level = level

        for i in range(level - 1):
            self.decoder.append(
                nn.Sequential(
                    (
                        BlurPool(in_ch1[i], pad_type="zero", stride=1, pad_off=1)
                        if down > 1
                        else nn.Identity()
                    ),
                    spectral_norm(
                        nn.Conv2d(
                            in_ch1[i],
                            out_ch,
                            kernel_size=3,
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
                )
            )

        self.decoder.append(
            nn.Sequential(spectral_norm(nn.Linear(in_ch2, out_ch)), activation)
        )
        self.out = spectral_norm(nn.Linear(out_ch, 1))
        self.embed = nn.Embedding(num_classes, out_ch) if num_classes > 0 else None

    def forward(self, x, c=None):
        final_pred = []
        for i in range(self.level - 1):
            final_pred.append(self.decoder[i](x[i]).squeeze(1))

        h = self.decoder[-1](x[-1].float())
        out = self.out(h)

        if self.embed is not None and c is not None:
            out += torch.sum(self.embed(c) * h, dim=1, keepdim=True)

        out = torch.tanh(out)
        final_pred.append(out)
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
        if self.use_freq:
            x = haar_highpass(x)
            x = (x - x.mean(dim=[2, 3], keepdim=True)) / (
                x.std(dim=[2, 3], keepdim=True) + 1e-6
            )
        else:
            x = x * 0.5 + 0.5
            x = (x - self.image_mean[:, None, None]) / self.image_std[:, None, None]

        features = self.model.encode_image(x, return_pooled_feats=True)
        features = self.decoder(features)

        loss_fn = multilevel_loss(alpha=self.gan_alpha)
        loss = loss_fn(features, for_real=for_real, for_G=for_G)

        if return_logits:
            return loss, features
        return loss


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
            self.encoder.append(
                nn.Sequential(
                    self._sn(nn.Conv2d(enc_chs[i], enc_chs[i + 1], 4, 2, 1)),
                    nn.GroupNorm(min(8, enc_chs[i + 1] // 4), enc_chs[i + 1]),
                    nn.LeakyReLU(0.2),
                    self._sn(nn.Conv2d(enc_chs[i + 1], enc_chs[i + 1], 3, 1, 1)),
                    nn.GroupNorm(min(8, enc_chs[i + 1] // 4), enc_chs[i + 1]),
                    nn.LeakyReLU(0.2),
                )
            )

        self.scale_heads = nn.ModuleList()
        for i in range(num_scales):
            ch = enc_chs[i + 1]
            self.scale_heads.append(
                nn.Sequential(
                    self._sn(nn.Conv2d(ch, ch // 2, 3, 1, 1)),
                    nn.LeakyReLU(0.2),
                    self._sn(nn.Conv2d(ch // 2, ch // 4, 3, 1, 1)),
                    nn.LeakyReLU(0.2),
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
        feats_image = self._extract_features(image)
        feats_ref = self._extract_features(ref)

        per_scale_logits = []
        for i, head in enumerate(self.scale_heads):
            diff = torch.abs(feats_image[i] - feats_ref[i])
            logit = head(diff)
            per_scale_logits.append(logit)

        weights = torch.softmax(self.scale_weights, dim=0)
        logits_stacked = torch.stack(per_scale_logits, dim=0)
        weighted = (weights.unsqueeze(1).unsqueeze(2) * logits_stacked).sum(dim=0)

        return torch.clamp(weighted, -5.0, 5.0), per_scale_logits


# ══════════════════════════════════════════════════════════════
#  SD2RefDiscriminator — LightningModule 封装
# ══════════════════════════════════════════════════════════════
class SD2RefDiscriminator(LightningModule):
    """
    双判别器封装：
      - 语义判别器 D_sem：ImageConvNextDiscriminator
      - 纹理判别器 D_tex：TextureConsistencyDiscriminator
    """

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

        if use_semantic_d:
            self.D_sem = ImageConvNextDiscriminator(
                alpha=semantic_alpha,
                precision=semantic_precision,
                use_freq=semantic_use_freq,
                trainable_stages=semantic_trainable_stages,
            )
        else:
            self.D_sem = None

        if use_texture_d:
            self.D_tex = TextureConsistencyDiscriminator(
                in_ch=3,
                base_ch=texture_base_ch,
                num_scales=texture_num_scales,
                use_spectral=texture_use_spectral,
            )
        else:
            self.D_tex = None

        self._d_sem_accum_count = 0
        self._d_tex_accum_count = 0

    def compute_g_loss(
        self,
        fake: torch.Tensor,
        ref: Optional[torch.Tensor] = None,
        lambda_semantic: float = 1.0,
        lambda_texture: float = 1.0,
    ) -> torch.Tensor:
        loss = 0.0

        if self.use_semantic_d and lambda_semantic > 0:
            loss_sem = self.D_sem(fake, for_real=False, for_G=True)
            loss = loss + lambda_semantic * loss_sem

        if self.use_texture_d and lambda_texture > 0 and ref is not None:
            fake_logit, _ = self.D_tex(fake, ref)
            loss = loss + lambda_texture * (-fake_logit.mean())

        return loss

    def compute_d_loss(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
        ref: Optional[torch.Tensor] = None,
        lambda_semantic: float = 1.0,
        lambda_texture: float = 1.0,
    ) -> torch.Tensor:
        loss = 0.0

        if self.use_semantic_d and lambda_semantic > 0:
            loss_sem_real = self.D_sem(real, for_real=True, for_G=False)
            loss_sem_fake = self.D_sem(fake.detach(), for_real=False, for_G=False)
            loss = loss + lambda_semantic * (loss_sem_real + loss_sem_fake)

        if self.use_texture_d and lambda_texture > 0 and ref is not None:
            real_logit, _ = self.D_tex(real, ref)
            fake_logit, _ = self.D_tex(fake.detach(), ref)
            loss_tex = F.relu(1.0 - real_logit).mean() + F.relu(1.0 + fake_logit).mean()
            loss = loss + lambda_texture * loss_tex

        return loss

    def forward(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
        ref: Optional[torch.Tensor] = None,
        mode: str = "both",
    ) -> Tuple[torch.Tensor, dict]:
        log_dict = {}

        if mode in ["both", "semantic"] and self.use_semantic_d:
            loss_sem = self.compute_d_loss(
                real, fake, ref=None, lambda_semantic=1.0, lambda_texture=0.0
            )
            log_dict["loss_D_sem"] = loss_sem.detach()

        if mode in ["both", "texture"] and self.use_texture_d and ref is not None:
            loss_tex = self.compute_d_loss(
                real, fake, ref=ref, lambda_semantic=0.0, lambda_texture=1.0
            )
            log_dict["loss_D_tex"] = loss_tex.detach()

        loss_D = sum(log_dict.values())
        log_dict["loss_D_total"] = loss_D.detach()

        return loss_D, log_dict

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        real = batch["hr"]
        fake = batch["sr"].detach()
        ref = batch.get("ref", None)

        if optimizer_idx == 0 and self.use_semantic_d:
            loss = self.compute_d_loss(
                real, fake, ref=None, lambda_semantic=1.0, lambda_texture=0.0
            )
            self.log("train/D_sem", loss, on_step=True, prog_bar=True)
            return loss

        if optimizer_idx == 1 and self.use_texture_d and ref is not None:
            loss = self.compute_d_loss(
                real, fake, ref=ref, lambda_semantic=0.0, lambda_texture=1.0
            )
            self.log("train/D_tex", loss, on_step=True, prog_bar=True)
            return loss

        if optimizer_idx is None:
            loss = self.compute_d_loss(real, fake, ref=ref)
            self.log("train/D_total", loss, on_step=True, prog_bar=True)
            return loss

        return None

    def configure_optimizers(self):
        opts = []

        if self.use_semantic_d:
            params_sem = [p for p in self.D_sem.parameters() if p.requires_grad]
            opts.append(
                torch.optim.AdamW(
                    params_sem,
                    lr=self.hparams.lr_semantic,
                    betas=self.hparams.betas,
                    weight_decay=self.hparams.weight_decay,
                )
            )

        if self.use_texture_d:
            params_tex = list(self.D_tex.parameters())
            opts.append(
                torch.optim.AdamW(
                    params_tex,
                    lr=self.hparams.lr_texture,
                    betas=self.hparams.betas,
                    weight_decay=self.hparams.weight_decay,
                )
            )

        return opts

    def on_save_checkpoint(self, checkpoint):
        checkpoint["d_sem_accum_count"] = self._d_sem_accum_count
        checkpoint["d_tex_accum_count"] = self._d_tex_accum_count

    def on_load_checkpoint(self, checkpoint):
        self._d_sem_accum_count = checkpoint.get("d_sem_accum_count", 0)
        self._d_tex_accum_count = checkpoint.get("d_tex_accum_count", 0)
