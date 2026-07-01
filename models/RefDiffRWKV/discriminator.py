# RefDiffRWKV/discriminator.py

import torch
from torch import nn
import open_clip
from vision_aided_loss.cv_discriminator import BlurPool, spectral_norm
from vision_aided_loss.cv_losses import multilevel_loss

import os

# ══════════════════════════════════════════════════════════════
#  Haar 小波高频分解（频域判别器核心）
# ══════════════════════════════════════════════════════════════


def haar_highpass(x):
    """
    Haar 小波高频分解：返回 [LH, HL, HH] 三个高频子带 concat。

    输入：(B, C, H, W)  RGB 像素图像
    输出：(B, C*3, H//2, W//2)  三个方向的高频细节

    LH: 水平边缘（左 - 右差异）
    HL: 垂直边缘（上 - 下差异）
    HH: 对角边缘/角点
    """
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
#  Backbone: OpenCLIP ConvNeXt
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
    """
    OpenCLIP ConvNeXt-XXLarge 视觉 backbone 封装。

    Args:
        precision:        "fp32" / "fp16"
        trainable_stages: 解冻 stem + 前 N 个 stage
                          0 = 全冻结（原行为）
                          1 = 解冻 stem（第一层卷积 + LayerNorm）
                          2 = 解冻 stem + stage1
                          3 = 解冻 stem + stage1 + stage2
    """

    def __init__(self, precision="fp32", trainable_stages=0):
        super().__init__()

        local_ckpt = os.path.expanduser(
            "~/.cache/huggingface/hub/"
            "models--laion--CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg-soup/"
            "snapshots/9f3e8ee3f383c672388d9178afe70af9e63ac9df/"
            "open_clip_pytorch_model.bin"
        )

        full_model, _, _ = open_clip.create_model_and_transforms(
            "convnext_xxlarge",
            pretrained=local_ckpt,
            precision=precision,
        )
        self.model = full_model.visual
        self.trainable_stages = trainable_stages

        # 按需解冻
        self.model.eval().requires_grad_(False)
        if trainable_stages >= 1:
            self.model.trunk.stem.requires_grad_(True)
        if trainable_stages >= 2:
            self.model.trunk.stages[0].requires_grad_(True)
        if trainable_stages >= 3:
            self.model.trunk.stages[1].requires_grad_(True)

    def encode_image(self, image, return_feats=False, return_pooled_feats=False):
        return _visual_forward(
            self.model,
            image,
            return_feats,
            return_pooled_feats,
        )


# ══════════════════════════════════════════════════════════════
#  MultiLevelDConv — 多尺度判别器头部（加深版）
# ══════════════════════════════════════════════════════════════


class MultiLevelDConv(nn.Module):

    def __init__(
        self,
        level=3,
        in_ch1=(384, 768, 1536),
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
                    # ── 第一层 3×3 ──
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
                    # ── 第二层 3×3 ──
                    spectral_norm(nn.Conv2d(out_ch, out_ch, 3, padding=1)),
                    activation,
                    # ═══════════════════════════════════════════
                    BlurPool(out_ch, pad_type="zero", stride=1),
                    spectral_norm(nn.Conv2d(out_ch, 1, kernel_size=1, stride=2)),
                    nn.Tanh(),  # ← 值域约束 [-1, 1]，防止空间预测无界发散
                )
            )
        self.decoder.append(
            nn.Sequential(spectral_norm(nn.Linear(in_ch2, out_ch)), activation)
        )
        self.out = spectral_norm(nn.Linear(out_ch, 1))
        self.embed = None
        if num_classes > 0:
            self.embed = nn.Embedding(num_classes, out_ch)

    def forward(self, x, c=None):
        final_pred = []
        for i in range(self.level - 1):
            final_pred.append(self.decoder[i](x[i]).squeeze(1))
        h = self.decoder[-1](x[-1].float())
        out = self.out(h)

        if self.embed is not None:
            out += torch.sum(self.embed(c) * h, 1, keepdim=True)

        out = torch.tanh(out)  # ← 值域约束 [-1, 1]，防止标量预测无界发散
        final_pred.append(out)
        return final_pred


# ══════════════════════════════════════════════════════════════
#  ImageConvNextDiscriminator — 语义判别器（频域 + 可训练 stem）
# ══════════════════════════════════════════════════════════════


class ImageConvNextDiscriminator(nn.Module):
    """
    语义级判别器：判断"这张图看起来真实吗"。

    采用 Haar 小波频域输入 + ConvNeXt-XXLarge backbone + 多尺度头部。
    主要用于保证生成图的低频合理性（色彩、语义一致性），
    纹理细节的驱动由 TextureConsistencyDiscriminator 负责。
    """

    def __init__(self, alpha=0.8, precision="fp32", use_freq=True, trainable_stages=1):
        """
        Args:
            alpha:             GAN loss 中 multilevel loss 的混合系数
            precision:         CLIP backbone 精度（"fp32" / "fp16"）
            use_freq:          True=频域判别（小波高频子带），False=像素域判别
            trainable_stages:  解冻 backbone 的层数
                               0 = 全冻结（原行为，不推荐）
                               1 = 解冻 stem（推荐起步值）
                               2 = 解冻 stem + stage1
        """
        super().__init__()
        self.gan_alpha = alpha
        self.use_freq = use_freq
        self.trainable_stages = trainable_stages

        self.model = ImageOpenCLIPConvNext(
            precision=precision,
            trainable_stages=trainable_stages,
        )

        self.decoder = MultiLevelDConv(
            level=3, in_ch1=[768, 1536], in_ch2=1024, out_ch=512, down=2
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
        """
        重新初始化 ConvNeXt 第一层卷积以适配 9 通道频域输入。
        前 3 通道复用 CLIP 预训练权重，后 6 通道用正交初始化。
        stem 的 requires_grad 由 trainable_stages 控制。
        """
        old_conv = self.model.model.trunk.stem[0]
        old_weight = old_conv.weight.data

        new_conv = nn.Conv2d(
            in_channels,
            old_weight.shape[0],
            kernel_size=4,
            stride=4,
            padding=0,
        )

        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = old_weight
            for c in range(3, in_channels):
                nn.init.orthogonal_(new_conv.weight[:, c : c + 1, :, :])
        new_conv.bias.data.copy_(old_conv.bias.data)

        new_conv.requires_grad_(self.trainable_stages >= 1)

        self.model.model.trunk.stem[0] = new_conv

        if self.trainable_stages >= 1:
            self.model.model.trunk.stem[1].requires_grad_(True)

    def train(self, mode=True):
        """只让 decoder + 已解冻的 stem/stage 进入 train 模式。"""
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

    def forward(
        self, x, for_real=True, for_G=False, verbose=False, return_logits=False
    ):
        if self.use_freq:
            x = haar_highpass(x)
            # 逐样本 z-score 标准化（高频子带是零均值差分，不用 CLIP 均值/方差）
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
#  TextureConsistencyDiscriminator — 纹理一致性判别器（遥感 RefSR 专用）
# ══════════════════════════════════════════════════════════════


class TextureConsistencyDiscriminator(nn.Module):
    """
    多尺度特征差值纹理一致性判别器 — 遥感 RefSR 专用。

    设计原理：
      不判断"生成图是否等于 ref"，而是判断"生成图相对 ref 的纹理差异
      是否和 HR 相对 ref 的纹理差异属于同一模式"。

      输入：(生成图或HR, ref)
      过程：共享编码器提取多尺度特征 → 计算特征差值 → 判别器判断差值模式
      输出：单个 logit（Hinge loss 用）

    为什么适合时间跨度的遥感场景：
      - 地物真实变化（农田→住宅）：差值大但纹理完整 → 判别器学到这是正常模式
      - 纹理迁移失败（模糊）：差值中高频子带能量异常低 → 被识别为异常
      - 纹理迁移错误（森林纹理错贴到城市）：差值空间分布异常 → 被识别
      - 小树长成大树：差值模式与"模糊"不同 → 不会被错判

    训练策略：
      - 正样本：(HR, ref) → 期望 logit > +1
      - 负样本：(生成图, ref) → 期望 logit < -1
      - 判别器学习的是"差值模式的统计分布"，而非"差值是否为零"

    参数：
      in_ch:        输入通道数（3 for RGB）
      base_ch:      基础通道数，~3M 总参数量
      num_scales:   多尺度层数（4 = 感受野从局部到全局覆盖完整）
      use_spectral: 是否使用 spectral norm（稳定训练）
    """

    def __init__(self, in_ch=3, base_ch=48, num_scales=4, use_spectral=True):
        super().__init__()
        self.num_scales = num_scales
        self._sn = spectral_norm if use_spectral else lambda x: x

        # ═══════════════════════════════════════════
        # 共享编码器（对 image 和 ref 使用同一套权重）
        # 逐层下采样 → 多尺度特征金字塔
        # ═══════════════════════════════════════════
        enc_chs = [in_ch] + [base_ch * (2**i) for i in range(num_scales)]
        # enc_chs: [3, 48, 96, 192, 384]

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

        # ═══════════════════════════════════════════
        # 多尺度差值判别器头部
        # 每个尺度独立处理差值特征后聚合
        # ═══════════════════════════════════════════
        self.scale_heads = nn.ModuleList()
        for i in range(num_scales):
            ch = enc_chs[i + 1]
            self.scale_heads.append(
                nn.Sequential(
                    # 差值特征 → 判别特征（两层 3×3 卷积）
                    self._sn(nn.Conv2d(ch, ch // 2, 3, 1, 1)),
                    nn.LeakyReLU(0.2),
                    self._sn(nn.Conv2d(ch // 2, ch // 4, 3, 1, 1)),
                    nn.LeakyReLU(0.2),
                    # 全局平均池化 → 单个判别分数
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    self._sn(nn.Linear(ch // 4, 1)),
                )
            )

        # ═══════════════════════════════════════════
        # 可学习的尺度聚合权重
        # 初始化为均匀分布，训练中自适应调整
        # ═══════════════════════════════════════════
        self.scale_weights = nn.Parameter(torch.ones(num_scales) / num_scales)

    def _extract_features(self, x):
        """提取多尺度特征金字塔。"""
        feats = []
        for layer in self.encoder:
            x = layer(x)
            feats.append(x)
        return feats  # [feat_s0(H×W), feat_s1(H/2×W/2), ..., feat_s{n-1}]

    def forward(self, image, ref):
        """
        Args:
            image: (B, 3, H, W)  生成图 或 HR 真实图
            ref:   (B, 3, H, W)  参考图像（已 resize 到相同尺寸）

        Returns:
            logit:           (B, 1)  真假评分（Hinge loss 用）
            per_scale_logits: List[(B, 1)]  各尺度 logit（用于调试/可视化）
        """
        # Step 1: 共享编码器提取多尺度特征
        feats_image = self._extract_features(image)
        feats_ref = self._extract_features(ref)

        # Step 2: 计算各尺度特征差值 → 判别器头部打分
        per_scale_logits = []
        for i, head in enumerate(self.scale_heads):
            diff = torch.abs(feats_image[i] - feats_ref[i])
            logit = head(diff)  # (B, 1)
            per_scale_logits.append(logit)

        # Step 3: 可学习加权聚合
        weights = torch.softmax(self.scale_weights, dim=0)
        logits_stacked = torch.stack(per_scale_logits, dim=0)  # (S, B, 1)
        weighted = (weights.unsqueeze(1).unsqueeze(2) * logits_stacked).sum(dim=0)

        return torch.tanh(weighted), per_scale_logits  # ← 值域约束 [-1, 1]
