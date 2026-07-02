# RefDiffRWKV/discriminator.py

import torch
from torch import nn
import open_clip
from vision_aided_loss.cv_discriminator import BlurPool, spectral_norm
from vision_aided_loss.cv_losses import multilevel_loss

import os


def haar_highpass(x):
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

        local_ckpt = os.path.expanduser(
            "~/.cache/huggingface/hub/models--laion--CLIP-convnext_base_w-laion2B-s13B-b82K/snapshots/main/open_clip_pytorch_model.bin"
        )

        full_model, _, _ = open_clip.create_model_and_transforms(
            "convnext_base_w",
            pretrained=local_ckpt,
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
        in_ch1=(384, 768),
        in_ch2=512,
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
#  ImageConvNextDiscriminator
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
            level=3, in_ch1=[384, 768], in_ch2=512, out_ch=256, down=2
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

    # train / eval / requires_grad_ 方法保持不变
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
