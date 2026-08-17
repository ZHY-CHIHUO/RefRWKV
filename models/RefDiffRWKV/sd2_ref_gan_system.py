"""
sd2_ref_gan_system.py — G/D 分离 + 交替训练系统

设计原则：
1. 持有 SD2RefGenerator 和 SD2RefDiscriminator；
2. 手动优化 + AMP + 梯度累积，按 phase 控制 G/D 交替；
3. G step 中扩散 loss 为主，diff_sr / LPIPS / GAN 为辅助；
4. D step 中用单步 pred_x0 生成 fake/real，更新判别器；
5. 所有进入判别器的图像统一保持在 [-1, 1] 值域。
6. 无 adapter 路径复用 generator UNet + 零残差注入，零额外显存开销。

四阶段训练：
- Stage 1  基础扩散：HR/SR 双路径噪声 MSE，SR 路径降频 + 权重递增
- Stage 2  语义注入：DINOv2(冻结) + RWKV 金字塔 → cross-attention，
           语义分支 5× 学习率分组
- Stage 3  像素感知约束：SR 起点单步重建，diff_sr 走 latent MSE，
           LPIPS 降频 + 半分辨率；t 收窄 [100, 400] 对齐推理 t_start=300
- Stage 4  对抗训练：随机切割 256×256 patch 喂 D_sem/D_tex，
           与 LPIPS 步错开执行，避免显存峰值叠加
"""

import os
import logging
import shutil
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
import lpips
import numpy as np
from PIL import Image

from .sd2_ref_generator import SD2RefGenerator
from .sd2_ref_discriminator import SD2RefDiscriminator

logger = logging.getLogger(__name__)


class SD2RefGANSystem(LightningModule):
    def __init__(
        self,
        generator: SD2RefGenerator,
        discriminator: Optional[SD2RefDiscriminator] = None,
        lambda_gan_semantic: float = 0.3,
        lambda_gan_texture: float = 0.5,
        lambda_lpips: float = 0.3,
        lambda_diff_sr: float = 0.5,
        accumulate_grad_batches: int = 8,
        use_amp: bool = True,
        g_d_ratio: int = 1,
        g_lr: float = 1e-4,
        g_weight_decay: float = 1e-3,
        d_lr_sem: float = 5e-6,
        d_lr_tex: float = 1e-6,
        d_weight_decay: float = 1e-3,
        betas: tuple = (0.5, 0.999),
        sample_steps: int = 50,
        fr_metrics: Optional[List[str]] = None,
        sr_model: Optional[torch.nn.Module] = None,
        sr_fixed: bool = True,
        sr_lr: float = 1e-5,
        t_start: Optional[int] = None,
        guidance_scale: float = 0.0,
        t_stop: int = 200,
        grad_clip_val: float = 1.0,
        grad_warn_threshold: float = 100.0,
        max_consecutive_nan: int = 10,
        gan_enabled: bool = False,
        use_swap_test: bool = False,
        swap_ratio: float = 0.5,
        dtex_conf_weight: bool = False,
        lambda_sr_noise: float = 1.0,
        sr_noise_warmdown_start: float = 1.0,
        sr_noise_warmdown_steps: int = 0,
        gan_crop_size: int = 256,
        train_t_min: int = 0,
        train_t_max: int = 999,
        aux_t_min: int = 100,
        aux_t_max: int = 400,
        gan_warmup_steps: int = 3000,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["generator", "discriminator", "sr_model"])

        self.generator = generator
        self.discriminator = discriminator
        self.sr_model = sr_model

        self.lambda_gan_semantic = lambda_gan_semantic
        self.lambda_gan_texture = lambda_gan_texture
        self.lambda_lpips = lambda_lpips
        self.lambda_diff_sr = lambda_diff_sr

        self.accumulate_grad_batches = accumulate_grad_batches
        self.sample_steps = sample_steps
        self.g_d_ratio = g_d_ratio

        self.t_start = t_start
        self.guidance_scale = guidance_scale
        self.t_stop = t_stop

        self.grad_clip_val = grad_clip_val
        self.grad_warn_threshold = grad_warn_threshold
        self.max_consecutive_nan = max_consecutive_nan

        self.sr_fixed = sr_fixed
        self.sr_lr = sr_lr

        self.gan_enabled = gan_enabled and discriminator is not None

        self.use_swap_test = use_swap_test
        self.swap_ratio = swap_ratio

        # D_tex 置信加权：开启后 D_tex 只在 raw cos_map
        # 高置信的局部匹配区域执法）
        self.dtex_conf_weight = dtex_conf_weight

        if lambda_sr_noise < 0:
            raise ValueError("lambda_sr_noise must be >= 0")
        if sr_noise_warmdown_start < 0:
            raise ValueError("sr_noise_warmdown_start must be >= 0")
        if sr_noise_warmdown_steps < 0:
            raise ValueError("sr_noise_warmdown_steps must be >= 0")
        self.lambda_sr_noise = lambda_sr_noise
        self.sr_noise_warmdown_start = sr_noise_warmdown_start
        self.sr_noise_warmdown_steps = sr_noise_warmdown_steps
        self.gan_crop_size = gan_crop_size
        assert train_t_min <= train_t_max, (
            f"train_t_min({train_t_min}) 必须 <= train_t_max({train_t_max})"
        )
        assert aux_t_min <= aux_t_max, (
            f"aux_t_min({aux_t_min}) 必须 <= aux_t_max({aux_t_max})"
        )
        self.train_t_min = train_t_min
        self.train_t_max = train_t_max
        self.aux_t_min = aux_t_min
        self.aux_t_max = aux_t_max
        self.gan_warmup_steps = gan_warmup_steps

        self._nan_g_count = 0
        self._nan_d_count = 0
        self._consecutive_nan_g = 0
        self._consecutive_nan_d = 0

        self.automatic_optimization = False

        self.use_amp = use_amp
        self.scaler_g = None

        self._g_accum_count = 0
        self._d_sem_accum_count = 0
        self._d_tex_accum_count = 0
        self._gd_phase = 0
        self._g_steps_since_d = 0
        self._g_optimizer_steps = 0
        self._opt_idx: dict = {}

        # LPIPS（延迟初始化：首次实际使用时才加载 VGG 权重）
        self.net_lpips = None

        # IQA
        self.iqa = None
        try:
            from RefRWKV.evaluation.eval_pyiqa import IQAEngine

            full_fr_metrics = fr_metrics or ["psnr", "ssim", "lpips", "dists"]
            iqa_fr_metrics = [m for m in full_fr_metrics if m != "lpips"]
            self.iqa = IQAEngine(
                device="cuda" if torch.cuda.is_available() else "cpu",
                nr_metrics=[],
                fr_metrics=iqa_fr_metrics,
                use_y_channel=True,
                verbose=False,
            )
        except (ImportError, RuntimeError) as e:
            logger.warning("IQA engine 不可用: %s", e)

    # ═══════════════════════════════════════════════════════
    #  Discriminator 冻结 / 解冻
    # ═══════════════════════════════════════════════════════

    def _get_lpips(self):
        """按需初始化并返回 VGG-LPIPS（延迟加载权重）。"""
        if self.net_lpips is None:
            self.net_lpips = lpips.LPIPS(net="vgg", verbose=False).to(self.device)
            for p in self.net_lpips.parameters():
                p.requires_grad = False
        return self.net_lpips

    def _freeze_discriminator(self):
        if self.discriminator is not None:
            self.discriminator.eval()
            self.discriminator.requires_grad_(False)

    def _unfreeze_discriminator(self):
        if self.discriminator is not None:
            self.discriminator.train()
            self.discriminator.requires_grad_(True)

    # ═══════════════════════════════════════════════════════
    #  构建全零 down_intrablock 残差（优化：使用 new_zeros）
    # ═══════════════════════════════════════════════════════

    def _build_zero_intrablock(self, x_input: torch.Tensor) -> List[torch.Tensor]:
        return self.generator.build_zero_intrablock(x_input)

    # ═══════════════════════════════════════════════════════
    #  公共 pred_x0 基础逻辑
    # ═══════════════════════════════════════════════════════

    def _pred_x0_base(
        self,
        latent,
        sr_latent_cond,
        t,
        noise,
        context,
        down_intrablock=None,
        return_pixel: bool = True,
    ):
        x_t = self.generator.add_noise(latent, noise, t)
        x_input = self.generator.concat_sr_latent(x_t, sr_latent_cond)

        eps_pred = self.generator.forward_unet(
            x_input,
            t,
            context,
            down_intrablock=down_intrablock,
        )

        pred_x0 = self.generator.predict_x0_from_eps(x_t, t, eps_pred)
        pred_x0 = torch.nan_to_num(pred_x0, nan=0.0, posinf=20.0, neginf=-20.0).clamp(
            -20.0, 20.0
        )
        if not return_pixel:
            return pred_x0
        return self.generator.decode_latent(pred_x0)

    # ═══════════════════════════════════════════════════════
    #  无 adapter 路径（用于生成 real target，始终 no_grad）
    # ═══════════════════════════════════════════════════════

    @torch.no_grad()
    def _no_adapter_pred_x0(self, hr, sr_latent_cond, t, noise):
        bsz = hr.shape[0]
        hr_latent = self.generator.encode_latent(hr)

        null_ctx = torch.zeros(
            bsz,
            self.generator.CROSS_ATTN_CTX_LEN,
            self.generator.cross_attn_dim,
            device=hr.device,
            dtype=torch.float32,
        )

        return self._pred_x0_base(
            latent=hr_latent,
            sr_latent_cond=(
                sr_latent_cond.detach() if sr_latent_cond is not None else None
            ),
            t=t,
            noise=noise,
            context=null_ctx,
            down_intrablock=None,
        )

    # ═══════════════════════════════════════════════════════
    #  有 adapter 路径（G step Phase2 需要梯度穿透到 UNet/Adapter）
    # ═══════════════════════════════════════════════════════

    def _adapter_pred_x0(
        self,
        lr,
        ref,
        sr_latent_precomputed,
        t,
        noise,
        return_conf: bool = False,
        return_latent: bool = False,
    ):
        """adapter 路径单步 pred_x0。

        Args:
            return_conf: True 时额外返回 conf_dtex——raw cos_map 的
                scale2 (B,1,60,60)，传播前的局部匹配置信，已 detach，
                供 D_tex 置信加权使用。
            return_latent: True 时返回 latent 空间的 pred_x0，不做 VAE decode。

        Returns:
            pred 或 (pred, conf_dtex)
        """
        cond = self.generator.prepare_condition(
            lr,
            ref,
            sr_latent=sr_latent_precomputed.detach(),
            t=t,
            return_conf=return_conf,
            latent_hw=sr_latent_precomputed.shape[-2:],
        )
        pred = self._pred_x0_base(
            latent=sr_latent_precomputed,
            sr_latent_cond=sr_latent_precomputed.detach(),
            t=t,
            noise=noise,
            context=cond["context"],
            down_intrablock=cond["down_intrablock"],
            return_pixel=not return_latent,
        )

        if return_conf:
            return pred, cond.get("conf_dtex")
        return pred

    # ═══════════════════════════════════════════════════════
    #  优化器 / 状态持久化
    # ═══════════════════════════════════════════════════════

    def _get_opt(self, key: str):
        """安全获取优化器（兼容 PL 单优化器返回对象而非列表的行为）。"""
        opts = self.optimizers()
        if not isinstance(opts, list):
            return opts if key == "g" else None
        idx = self._opt_idx.get(key)
        return opts[idx] if idx is not None else None

    def _get_g_opt(self):
        return self._get_opt("g")

    def _get_d_sem_opt(self):
        return self._get_opt("d_sem")

    def _get_d_tex_opt(self):
        return self._get_opt("d_tex")

    def _get_sr_opt(self):
        return self._get_opt("sr")

    def configure_optimizers(self):
        opts = []

        # 收集 Generator 可训练参数，排除 sr_model（由独立 SR 优化器管理）
        sr_param_ids = set()
        if self.sr_model is not None:
            sr_param_ids = {id(p) for p in self.sr_model.parameters()}

        g_params = [
            p
            for p in self.generator.parameters()
            if p.requires_grad and id(p) not in sr_param_ids
        ]

        if not g_params:
            logger.warning("Generator 无可训练参数，使用占位参数")
            if not hasattr(self, "_g_dummy_param"):
                self._g_dummy_param = nn.Parameter(torch.zeros(1))
            g_params = [self._g_dummy_param]

        # 语义分支（随机初始化）用 5 倍学习率，LoRA/Adapter 保持原学习率
        new_semantic_params = []
        pretrained_params = []
        for n, p in self.generator.named_parameters():
            if not p.requires_grad or id(p) in sr_param_ids:
                continue
            if any(k in n for k in ["semantic_pyramid", "sem_proj", "sr_conditioner"]):
                new_semantic_params.append(p)
            else:
                pretrained_params.append(p)

        if new_semantic_params:
            g_opt = torch.optim.AdamW(
                [
                    {"params": pretrained_params, "lr": self.hparams.g_lr},
                    {
                        "params": new_semantic_params,
                        "lr": self.hparams.g_lr * 5,
                        "semantic_group": True,  # 标记，_override_lr_on_resume 据此保留 5× lr
                    },
                ],
                weight_decay=self.hparams.g_weight_decay,
            )
            logger.info(
                "G 优化器参数分组: pretrained=%d (lr=%.1e), semantic=%d (lr=%.1e)",
                len(pretrained_params),
                self.hparams.g_lr,
                len(new_semantic_params),
                self.hparams.g_lr * 5,
            )
        else:
            g_opt = torch.optim.AdamW(
                g_params,
                lr=self.hparams.g_lr,
                weight_decay=self.hparams.g_weight_decay,
            )
        self._opt_idx["g"] = len(opts)
        opts.append(g_opt)

        # SR 模型优化器（仅 sr_fixed=False 时）
        if (
            self.sr_model is not None
            and not self.sr_fixed
            and any(p.requires_grad for p in self.sr_model.parameters())
        ):
            sr_params = [p for p in self.sr_model.parameters() if p.requires_grad]
            sr_opt = torch.optim.AdamW(
                sr_params,
                lr=self.hparams.sr_lr,
                weight_decay=self.hparams.g_weight_decay,
            )
            self._opt_idx["sr"] = len(opts)
            opts.append(sr_opt)
            logger.info(
                "SR 优化器已创建 (lr=%.1e, params=%d)",
                self.hparams.sr_lr,
                sum(p.numel() for p in sr_params),
            )

        # Discriminator 优化器
        if self.discriminator is not None:
            if self.discriminator.use_semantic_d:
                ps = [
                    p for p in self.discriminator.D_sem.parameters() if p.requires_grad
                ]
                if ps:
                    d_opt = torch.optim.AdamW(
                        ps,
                        lr=self.hparams.d_lr_sem,
                        betas=self.hparams.betas,
                        weight_decay=self.hparams.d_weight_decay,
                    )
                    self._opt_idx["d_sem"] = len(opts)
                    opts.append(d_opt)
                else:
                    logger.warning("D_sem 无可训练参数，跳过")

            if self.discriminator.use_texture_d:
                ps = [
                    p for p in self.discriminator.D_tex.parameters() if p.requires_grad
                ]
                if ps:
                    d_opt = torch.optim.AdamW(
                        ps,
                        lr=self.hparams.d_lr_tex,
                        betas=self.hparams.betas,
                        weight_decay=self.hparams.d_weight_decay,
                    )
                    self._opt_idx["d_tex"] = len(opts)
                    opts.append(d_opt)
                else:
                    logger.warning("D_tex 无可训练参数，跳过")

        return opts

    # ═══════════════════════════════════════════════════════
    #  Checkpoint 持久化
    # ═══════════════════════════════════════════════════════

    def _effective_lambda_sr_noise(self) -> float:
        """按已完成的 G optimizer step 线性退火 SR noise loss 权重。"""
        if self.sr_noise_warmdown_steps == 0:
            return self.lambda_sr_noise
        progress = min(
            self._g_optimizer_steps / self.sr_noise_warmdown_steps,
            1.0,
        )
        return self.sr_noise_warmdown_start + progress * (
            self.lambda_sr_noise - self.sr_noise_warmdown_start
        )

    def on_save_checkpoint(self, checkpoint):
        checkpoint.update(
            {
                "gd_phase": self._gd_phase,
                "g_accum_count": self._g_accum_count,
                "d_sem_accum_count": self._d_sem_accum_count,
                "d_tex_accum_count": self._d_tex_accum_count,
                "g_steps_since_d": self._g_steps_since_d,
                "g_optimizer_steps": self._g_optimizer_steps,
            }
        )

    def on_load_checkpoint(self, checkpoint):
        # 强制归零：checkpoint 不保存 .grad，非边界恢复会导致提前 step
        saved_g = checkpoint.get("g_accum_count", 0)
        if saved_g != 0:
            logger.warning(
                "从非累积边界恢复 (g_accum=%d)，梯度已丢失，计数器归零", saved_g
            )
        self._gd_phase = 0
        self._g_accum_count = 0
        self._d_sem_accum_count = 0
        self._d_tex_accum_count = 0
        self._g_steps_since_d = 0
        self._g_optimizer_steps = checkpoint.get("g_optimizer_steps", 0)

        # 参数分组导致 optimizer 组数不匹配时丢弃 optimizer 状态
        # 注意：on_load_checkpoint 执行时 optimizer 尚未配置，self.optimizers()
        # 拿不到参数组；改为按 generator 结构推导 G 优化器应有的组数
        # （与 scripts/train_sd2_gan.py 预检测逻辑一致）。
        opt_states = checkpoint.get("optimizer_states")
        if opt_states:
            saved_groups = len(opt_states[0].get("param_groups", []))
            gen = self.generator
            has_semantic_group = any(
                p.requires_grad
                and any(
                    k in n for k in ("semantic_pyramid", "sem_proj", "sr_conditioner")
                )
                for n, p in gen.named_parameters()
            )
            expected_groups = 2 if has_semantic_group else 1

            if saved_groups != expected_groups:
                logger.warning(
                    "optimizer param_groups 不匹配（checkpoint=%d, 预期=%d），"
                    "丢弃 optimizer/lr_scheduler 状态，仅恢复模型权重",
                    saved_groups,
                    expected_groups,
                )
                checkpoint.pop("optimizer_states", None)
                checkpoint.pop("lr_schedulers", None)
            else:
                logger.info(
                    "optimizer param_groups 匹配（%d 组），正常恢复 optimizer 状态",
                    saved_groups,
                )

    def load_state_dict(self, state_dict, strict=True):
        skip_prefix = "generator.global_semantic.semantic_pyramid."

        # 提取 pyramid 相关 keys
        pyramid_keys = [k for k in state_dict if k.startswith(skip_prefix)]

        # 当前 WKV4 公式：同时存在 key 和 receptance 线性层
        has_key = any("key.weight" in k for k in pyramid_keys)
        has_receptance = any("receptance.weight" in k for k in pyramid_keys)
        is_new_formula = has_key and has_receptance

        if pyramid_keys and not is_new_formula:
            logger.info(
                "跳过 %d 个不兼容的 semantic_pyramid 权重（WKV 公式不一致）",
                len(pyramid_keys),
            )
            state_dict = {
                k: v for k, v in state_dict.items() if not k.startswith(skip_prefix)
            }
            strict = False
        elif pyramid_keys:
            logger.info(
                "semantic_pyramid 权重为新公式，正常加载（%d 个 keys）",
                len(pyramid_keys),
            )

        # Stage1 → Stage2 过渡
        if self.generator.global_semantic is not None:
            has_semantic_in_ckpt = any(
                k.startswith("generator.global_semantic.") for k in state_dict
            )
            if not has_semantic_in_ckpt:
                logger.info(
                    "Checkpoint 无 global_semantic 权重（Stage1→2 过渡），"
                    "DINOv2 从预训练加载，pyramid/proj 随机初始化"
                )
                strict = False

        # 跨阶段恢复（如 Stage3→4）时 checkpoint 无 discriminator 权重
        if self.discriminator is not None:
            disc_keys_in_ckpt = [
                k for k in state_dict if k.startswith("discriminator.")
            ]
            disc_keys_expected = list(self.discriminator.state_dict().keys())
            if disc_keys_expected and not disc_keys_in_ckpt:
                logger.info(
                    "Discriminator 权重不在 checkpoint 中（%d keys），使用随机初始化",
                    len(disc_keys_expected),
                )
                strict = False

        # LPIPS 延迟初始化（优化后）：checkpoint 在验证时会写入 net_lpips 权重，
        # 而新进程加载时 net_lpips 尚未初始化（None），strict 恢复会把这些键判为
        # unexpected 而崩溃。此处直接丢弃这些键：_get_lpips() 首次验证时会从 lpips
        # 包重新加载同一份预训练 VGG 权重（同环境同版本），数值等价。
        if self.net_lpips is None:
            lpips_keys = [k for k in state_dict if k.startswith("net_lpips.")]
            if lpips_keys:
                logger.info(
                    "丢弃 %d 个 net_lpips 键（LPIPS 延迟初始化未触发，"
                    "首次验证时由 _get_lpips() 重新加载预训练权重）",
                    len(lpips_keys),
                )
                state_dict = {
                    k: v for k, v in state_dict.items()
                    if not k.startswith("net_lpips.")
                }

        result = super().load_state_dict(state_dict, strict=strict)
        missing, unexpected = result.missing_keys, result.unexpected_keys

        if missing:
            expected_new = (
                "semantic_pyramid",
                "sr_conditioner",
                "sim_transfer",
                "global_semantic",
                "sem_proj",
            )
            non_disc = [
                k
                for k in missing
                if not k.startswith("discriminator.")
                and not any(p in k for p in expected_new)
            ]
            if non_disc:
                logger.warning("load_state_dict: %d missing keys", len(non_disc))
                for k in non_disc[:10]:
                    logger.warning("  - %s", k)
            expected_missing = [
                k
                for k in missing
                if not k.startswith("discriminator.")
                and any(p in k for p in expected_new)
            ]
            if expected_missing:
                logger.info(
                    "load_state_dict: %d 个预期内新增参数随机初始化 "
                    "(global_semantic / sem_proj / pyramid / sr_conditioner)",
                    len(expected_missing),
                )
        if unexpected:
            logger.warning("load_state_dict: %d unexpected keys", len(unexpected))
            for k in unexpected[:10]:
                logger.warning("  - %s", k)
        if not missing and not unexpected:
            logger.info("load_state_dict: all keys matched")
        return result

    def _override_lr_on_resume(self):
        opts = self.optimizers()
        if not opts:
            return
        optimizers = opts if isinstance(opts, list) else [opts]

        # G 优化器：semantic 组（带 semantic_group 标记）保留 5× lr，
        # 其余组恢复为 g_lr（修复此前把所有组一律压成 g_lr 导致 5× 失效的 bug）
        for i, pg in enumerate(optimizers[self._opt_idx["g"]].param_groups):
            old_g = pg["lr"]
            new_lr = self.hparams.g_lr * (5.0 if pg.get("semantic_group") else 1.0)
            pg["lr"] = new_lr
            logger.info(
                "G LR[%d]: %.1e → %.1e%s",
                i, old_g, new_lr,
                " (semantic 5×)" if pg.get("semantic_group") else "",
            )

        sr_idx = self._opt_idx.get("sr")
        if sr_idx is not None:
            old_sr = None
            for pg in optimizers[sr_idx].param_groups:
                old_sr = pg["lr"]
                pg["lr"] = self.hparams.sr_lr
            if old_sr is not None:
                logger.info("SR LR: %.1e → %.1e", old_sr, self.hparams.sr_lr)

        for key, opt_key in [("d_sem", "d_lr_sem"), ("d_tex", "d_lr_tex")]:
            idx = self._opt_idx.get(key)
            if idx is not None:
                old = None
                for pg in optimizers[idx].param_groups:
                    old = pg["lr"]
                    pg["lr"] = getattr(self.hparams, opt_key)
                if old is not None:
                    logger.info(
                        "%s LR: %.1e → %.1e", key, old, getattr(self.hparams, opt_key)
                    )

    # ═══════════════════════════════════════════════════════
    #  梯度监控
    # ═══════════════════════════════════════════════════════

    def _monitor_grad_norms(self, optimizer, name: str):
        total_norm = (
            sum(
                p.grad.data.norm(2).item() ** 2
                for group in optimizer.param_groups
                for p in group["params"]
                if p.grad is not None
            )
            ** 0.5
        )
        if total_norm > self.grad_warn_threshold:
            logger.warning(
                "梯度爆炸警告 [%s]: grad_norm=%.2f > threshold=%.2f",
                name,
                total_norm,
                self.grad_warn_threshold,
            )
        return total_norm

    # ═══════════════════════════════════════════════════════
    #  SR latent 获取
    # ═══════════════════════════════════════════════════════

    @torch.no_grad()
    def _get_sr_latent_precomputed(self, lr, ref):
        """SR latent（无梯度，用于 D step 和 sr_fixed=True 的 G step）。"""
        if self.sr_model is None:
            return None
        with torch.amp.autocast(self.device.type, enabled=False):
            sr_pixel = self.sr_model(lr.float(), ref.float())
            sr_pixel = torch.nan_to_num(
                sr_pixel, nan=0.0, posinf=1.0, neginf=-1.0
            ).clamp(-1.0, 1.0)
            return self.generator.encode_latent(sr_pixel.to(self.generator.latent_dtype))

    def _get_sr_latent_with_grad(self, lr, ref):
        """SR latent（保留计算图，用于 sr_fixed=False 的 G step，梯度反传到 SR 模型）。

        绕过 _compute_sr_prior / encode_latent 的 no_grad，
        让梯度从 loss → UNet → latent → VAE → sr_pixel → SR 模型 完整反传。
        """
        if self.sr_model is None:
            return None
        with torch.amp.autocast(self.device.type, enabled=False):
            sr_pixel = self.sr_model(lr.float(), ref.float())
            sr_pixel = torch.nan_to_num(
                sr_pixel, nan=0.0, posinf=1.0, neginf=-1.0
            ).clamp(-1.0, 1.0)
            return self.generator.encode_latent_with_grad(
                sr_pixel.to(self.generator.latent_dtype)
            )

    # ═══════════════════════════════════════════════════════
    #  Stage 1/2: SR 路径噪声 MSE 计算
    # ═══════════════════════════════════════════════════════

    def _sample_train_timesteps(self, bsz: int, device, t_min=None, t_max=None):
        lo = self.train_t_min if t_min is None else t_min
        hi = self.train_t_max if t_max is None else t_max
        return torch.randint(lo, hi + 1, (bsz,), device=device, dtype=torch.long)

    def _compute_hr_noise_loss(self, lr, ref, hr_latent, sr_latent):
        """单路径 HR epsilon MSE，timestep 由 System 持有。"""
        bsz = lr.shape[0]
        device = lr.device
        noise = torch.randn_like(hr_latent)
        t = self._sample_train_timesteps(bsz, device)
        x_t = self.generator.add_noise(hr_latent, noise, t)
        x_input = self.generator.concat_sr_latent(x_t, sr_latent)
        cond = self.generator.prepare_condition(
            lr,
            ref,
            sr_latent=x_input[:, 4:].detach(),
            t=t,
            latent_hw=hr_latent.shape[-2:],
        )
        eps_pred = self.generator.forward_unet(
            x_input, t, cond["context"], cond["down_intrablock"]
        )
        return F.mse_loss(eps_pred, noise)

    def _compute_dual_path_loss(self, lr, ref, hr_latent, sr_latent):
        """HR/SR 双路径共享 noise + timestep，一次 UNet 前向。"""
        bsz = lr.shape[0]
        device = lr.device

        noise = torch.randn_like(hr_latent)
        t = self._sample_train_timesteps(bsz, device)

        x_t_hr = self.generator.add_noise(hr_latent, noise, t)
        x_t_sr = self.generator.add_noise(sr_latent, noise, t)

        x_t_combined = torch.cat([x_t_hr, x_t_sr], dim=0)
        t_combined = torch.cat([t, t], dim=0)
        sr_latent_combined = torch.cat([sr_latent, sr_latent], dim=0)

        cond = self.generator.prepare_condition(
            lr,
            ref,
            sr_latent=sr_latent.detach(),
            t=t,
            latent_hw=sr_latent.shape[-2:],
        )
        cond = self.generator.expand_condition(cond, n=2)

        x_input_combined = self.generator.concat_sr_latent(
            x_t_combined, sr_latent_combined
        )
        eps_pred_combined = self.generator.forward_unet(
            x_input_combined,
            t_combined,
            cond["context"],
            cond["down_intrablock"],
        )

        eps_pred_hr, eps_pred_sr = torch.chunk(eps_pred_combined, 2, dim=0)
        loss_hr = F.mse_loss(eps_pred_hr, noise)
        loss_sr = F.mse_loss(eps_pred_sr, noise)
        return loss_hr, loss_sr

    # ═══════════════════════════════════════════════════════
    #  Early Stop
    # ═══════════════════════════════════════════════════════

    def _check_early_stop(self, is_g_step: bool):
        cnt = self._consecutive_nan_g if is_g_step else self._consecutive_nan_d
        if cnt >= self.max_consecutive_nan:
            logger.error(
                "连续 %d 步 NaN (%s)，自动停止训练",
                self.max_consecutive_nan,
                "G step" if is_g_step else "D step",
            )
            self.trainer.should_stop = True
            return True
        return False

    # ═══════════════════════════════════════════════════════
    #  Training Step 路由
    # ═══════════════════════════════════════════════════════

    def training_step(self, batch, batch_idx):
        # D 预热：仅在 GAN 启用时
        if self.gan_enabled and self.global_step < self.gan_warmup_steps:
            return self._discriminator_step(batch, batch_idx)
        # G/D 交替（仅 GAN 启用时切 D phase）
        if self._gd_phase == 0:
            return self._generator_step(batch, batch_idx)
        return self._discriminator_step(batch, batch_idx)

    # ═══════════════════════════════════════════════════════
    #  Generator Step
    # ═══════════════════════════════════════════════════════

    def _generator_step(self, batch, batch_idx):
        g_opt = self._get_g_opt()
        self._freeze_discriminator()

        lr, ref, hr = self.generator.get_input(batch)

        # 在 try 块外初始化 sr_latent 和 hr_latent，供异常路径引用
        sr_latent = None
        hr_latent = None

        # ═══════════════════════════════════════════════════════
        # Phase 1: 二合一双路径扩散 ε-prediction loss
        # ═══════════════════════════════════════════════════════
        try:
            with torch.amp.autocast(
                self.device.type, enabled=self.use_amp, dtype=torch.bfloat16
            ):
                # 1. 编码 hr_latent
                hr_latent = self.generator.encode_latent(hr)

                # ★ 2. 根据 sr_fixed 选择是否保留梯度
                if not self.sr_fixed:
                    sr_latent = self._get_sr_latent_with_grad(lr, ref)  # 保留梯度
                else:
                    sr_latent = self._get_sr_latent_precomputed(lr, ref)  # 无梯度

                # 3. 双路径计算
                effective_lambda_sr_noise = self._effective_lambda_sr_noise()
                if sr_latent is not None and effective_lambda_sr_noise > 0:
                    loss_hr, loss_sr = self._compute_dual_path_loss(
                        lr, ref, hr_latent, sr_latent
                    )
                    loss = loss_hr + effective_lambda_sr_noise * loss_sr
                    self.log("train/G_hr_noise", loss_hr.detach(), on_step=True)
                    self.log("train/G_sr_noise", loss_sr.detach(), on_step=True)
                else:
                    # Stage 3+：关闭 SR epsilon，但仍由 System 采样 timestep
                    loss_hr = self._compute_hr_noise_loss(
                        lr, ref, hr_latent, sr_latent
                    )
                    loss = loss_hr
                    loss_sr = torch.zeros_like(loss)
                    self.log("train/G_hr_noise", loss_hr.detach(), on_step=True)
                    self.log("train/G_sr_noise", loss_sr.detach(), on_step=True)
                self.log(
                    "train/lambda_sr_noise_effective",
                    effective_lambda_sr_noise,
                    on_step=True,
                )

        except (RuntimeError, TypeError, AttributeError) as e:
            err_msg = str(e)
            is_cuda_error = isinstance(e, RuntimeError) and (
                "CUDA" in err_msg or "cuda" in err_msg or "CUBLAS" in err_msg
            )
            if is_cuda_error:
                logger.warning(
                    "[G step] 主扩散前向 CUDA 异常 (batch=%d): %s，跳过并重置",
                    batch_idx,
                    e,
                    exc_info=True,
                )
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                except RuntimeError:
                    pass
                try:
                    torch.cuda.synchronize()
                except RuntimeError:
                    logger.error("[G step] CUDA context 已损坏，无法恢复")
                    self.trainer.should_stop = True
                    return None
                g_opt.zero_grad(set_to_none=True)
                self._g_accum_count = 0
                if self.sr_model is not None:
                    try:
                        self.sr_model.to(self.device)
                    except RuntimeError:
                        pass
                return None
            raise

        # NaN 检查
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning("[G step] 主扩散 loss NaN/Inf, batch=%d", batch_idx)
            self._consecutive_nan_g += 1
            if self._check_early_stop(is_g_step=True):
                return None
            g_opt.zero_grad(set_to_none=True)
            self._g_accum_count = 0
            if self.gan_enabled:
                self._gd_phase = 1
            return None
        self._consecutive_nan_g = 0

        # 反向传播 Phase 1
        loss_main = loss / self.accumulate_grad_batches
        if self.scaler_g is not None:
            self.scaler_g.scale(loss_main).backward()
        else:
            loss_main.backward()

        # ═══════════════════════════════════════════════════════
        # Phase 2: 辅助 loss (diff_sr / LPIPS / GAN)
        # ═══════════════════════════════════════════════════════
        aux_loss_val = None
        aux_loss_enabled = self.sr_model is not None and (
            self.lambda_diff_sr > 0 or self.lambda_lpips > 0
        )
        gan_active = self.gan_enabled and (
            self.lambda_gan_semantic > 0 or self.lambda_gan_texture > 0
        )

        if aux_loss_enabled or gan_active:
            bsz = lr.shape[0]
            try:
                with torch.amp.autocast(
                    self.device.type, enabled=self.use_amp, dtype=torch.bfloat16
                ):
                    # ★ 获取 sr_latent (如果 Phase 1 没算，这里算)
                    if sr_latent is None:
                        if not self.sr_fixed:
                            sr_latent = self._get_sr_latent_with_grad(lr, ref)
                        else:
                            sr_latent = self._get_sr_latent_precomputed(lr, ref)

                    # Phase 2 的 t 范围由 System 持有（默认 [100, 400]，对齐推理 t_start=300）
                    t_sr = self._sample_train_timesteps(
                        bsz, lr.device, t_min=self.aux_t_min, t_max=self.aux_t_max
                    )
                    noise_sr = torch.randn_like(sr_latent)

                    # 调用 _adapter_pred_x0 获取 pred_x0_latent
                    if self.dtex_conf_weight:
                        pred_x0_latent, conf_dtex = self._adapter_pred_x0(
                            lr,
                            ref,
                            sr_latent,
                            t_sr,
                            noise_sr,
                            return_conf=True,
                            return_latent=True,
                        )
                    else:
                        pred_x0_latent = self._adapter_pred_x0(
                            lr,
                            ref,
                            sr_latent,
                            t_sr,
                            noise_sr,
                            return_latent=True,
                        )
                        conf_dtex = None

                    aux_loss = 0.0

                    # 1. diff_sr (Latent MSE)
                    if self.lambda_diff_sr > 0:
                        loss_diff_sr = F.mse_loss(pred_x0_latent, hr_latent.detach())
                        aux_loss = aux_loss + self.lambda_diff_sr * loss_diff_sr
                        self.log("train/G_diff_sr", loss_diff_sr.detach(), on_step=True)

                    # 2. LPIPS (降频，下采样)
                    pred_sr_pixel = None
                    if self.lambda_lpips > 0 and batch_idx % 4 == 0:
                        pred_sr_pixel = self.generator.decode_latent(pred_x0_latent)
                        pred_small = F.interpolate(
                            pred_sr_pixel,
                            scale_factor=0.5,
                            mode="bilinear",
                            align_corners=False,
                        )
                        hr_small = F.interpolate(
                            hr,
                            scale_factor=0.5,
                            mode="bilinear",
                            align_corners=False,
                        )
                        loss_lpips_sr = (
                            self._get_lpips()(pred_small, hr_small).mean()
                            * self.lambda_lpips
                        )
                        if not torch.isnan(loss_lpips_sr) and not torch.isinf(
                            loss_lpips_sr
                        ):
                            aux_loss = aux_loss + loss_lpips_sr
                            self.log(
                                "train/G_lpips", loss_lpips_sr.detach(), on_step=True
                            )
                        else:
                            self._nan_g_count += 1
                            logger.warning(
                                "[G step] LPIPS NaN/Inf (#%d)，跳过", self._nan_g_count
                            )

                    # 3. GAN Loss (降频，动态裁剪)
                    if gan_active and batch_idx % 4 == 2:
                        if pred_sr_pixel is None:
                            pred_sr_pixel = self.generator.decode_latent(pred_x0_latent)

                        H, W = pred_sr_pixel.shape[-2:]
                        crop_size = self.gan_crop_size
                        if H > crop_size and W > crop_size:
                            i = torch.randint(0, H - crop_size + 1, (1,)).item()
                            j = torch.randint(0, W - crop_size + 1, (1,)).item()
                            fake_crop = pred_sr_pixel[
                                :, :, i : i + crop_size, j : j + crop_size
                            ]
                            ref_crop = ref[:, :, i : i + crop_size, j : j + crop_size]

                            # 动态计算 conf 裁剪比例
                            scale_h = (
                                conf_dtex.shape[-2] / H if conf_dtex is not None else 1
                            )
                            scale_w = (
                                conf_dtex.shape[-1] / W if conf_dtex is not None else 1
                            )
                            conf_crop = (
                                conf_dtex[
                                    :,
                                    :,
                                    int(i * scale_h) : int((i + crop_size) * scale_h),
                                    int(j * scale_w) : int((j + crop_size) * scale_w),
                                ]
                                if conf_dtex is not None
                                else None
                            )
                        else:
                            fake_crop = pred_sr_pixel
                            ref_crop = ref
                            conf_crop = conf_dtex

                        with torch.amp.autocast(self.device.type, enabled=False):
                            gan_loss = self.discriminator.compute_g_loss(
                                fake_crop.float(),
                                ref=ref_crop.float(),
                                lambda_semantic=self.lambda_gan_semantic,
                                lambda_texture=self.lambda_gan_texture,
                                tex_weight=conf_crop,
                            )
                        if not torch.isnan(gan_loss) and not torch.isinf(gan_loss):
                            aux_loss = aux_loss + gan_loss
                            self.log("train/G_gan", gan_loss.detach(), on_step=True)
                        else:
                            self._nan_g_count += 1
                            logger.warning(
                                "[G step] GAN NaN/Inf (#%d)", self._nan_g_count
                            )

                    # 反向传播 Phase 2
                    if isinstance(aux_loss, torch.Tensor) and aux_loss.item() != 0:
                        aux_loss_val = aux_loss.detach()
                        aux_loss_scaled = aux_loss / self.accumulate_grad_batches
                        if self.scaler_g is not None:
                            self.scaler_g.scale(aux_loss_scaled).backward()
                        else:
                            aux_loss_scaled.backward()

            except (RuntimeError, TypeError, AttributeError) as e:
                err_msg = str(e)
                is_cuda_error = isinstance(e, RuntimeError) and (
                    "CUDA" in err_msg or "cuda" in err_msg or "CUBLAS" in err_msg
                )
                if is_cuda_error:
                    logger.warning(
                        "[G step] 辅助 loss CUDA/设备异常 (batch=%d): %s，跳过",
                        batch_idx,
                        e,
                        exc_info=True,
                    )
                    try:
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                    except RuntimeError:
                        pass
                    if self.sr_model is not None:
                        try:
                            self.sr_model.to(self.device)
                        except RuntimeError:
                            pass
                else:
                    raise

        # ═══════════════════════════════════════════════════════
        # 梯度累积 & Optimizer Step
        # ═══════════════════════════════════════════════════════
        self._g_accum_count += 1

        if self._g_accum_count >= self.accumulate_grad_batches:
            # 仅在 SR 模型有可训练参数且需要梯度时创建 sr_opt
            sr_has_grad = (
                not self.sr_fixed and sr_latent is not None and sr_latent.requires_grad
            )
            sr_opt = (
                self._get_sr_opt()
                if (sr_has_grad and (aux_loss_enabled or gan_active))
                else None
            )

            if self.scaler_g is not None:
                self.scaler_g.unscale_(g_opt)
                if sr_opt is not None:
                    self.scaler_g.unscale_(sr_opt)

            self._monitor_grad_norms(g_opt, "G")
            self.clip_gradients(
                g_opt,
                gradient_clip_val=self.grad_clip_val,
                gradient_clip_algorithm="norm",
            )
            if sr_opt is not None:
                self._monitor_grad_norms(sr_opt, "SR")
                self.clip_gradients(
                    sr_opt,
                    gradient_clip_val=self.grad_clip_val,
                    gradient_clip_algorithm="norm",
                )

            if self.scaler_g is not None:
                g_found_inf = self.scaler_g._found_inf_per_device(g_opt)
                g_optimizer_stepped = not any(
                    found_inf.item() != 0 for found_inf in g_found_inf.values()
                )
                self.scaler_g.step(g_opt)
                if sr_opt is not None:
                    self.scaler_g.step(sr_opt)
                self.scaler_g.update()
            else:
                g_opt.step()
                if sr_opt is not None:
                    sr_opt.step()
                g_optimizer_stepped = True

            g_opt.zero_grad(set_to_none=True)
            if sr_opt is not None:
                sr_opt.zero_grad(set_to_none=True)

            self._g_accum_count = 0
            if g_optimizer_stepped:
                self._g_optimizer_steps += 1
            self._g_steps_since_d += 1

            if self.gan_enabled and self._g_steps_since_d >= self.g_d_ratio:
                self._gd_phase = 1
                self._g_steps_since_d = 0
                self._unfreeze_discriminator()

        # ═══════════════════════════════════════════════════════
        # 日志记录
        # ═══════════════════════════════════════════════════════
        g_total = loss.detach() + (aux_loss_val if aux_loss_val is not None else 0.0)
        self.log("train/G_total", g_total, on_step=True, prog_bar=True)
        self.log("train/G_diff_hr", loss_hr.detach(), on_step=True, prog_bar=True)

        return g_total

    # ═══════════════════════════════════════════════════════
    #  Discriminator Step
    # ═══════════════════════════════════════════════════════
    def _discriminator_step(self, batch, batch_idx):
        if self.discriminator is None or (
            self.lambda_gan_semantic == 0.0 and self.lambda_gan_texture == 0.0
        ):
            self._gd_phase = 0
            return None

        if self.sr_model is None:
            self._gd_phase = 0
            return None

        self._unfreeze_discriminator()
        d_sem_opt, d_tex_opt = self._get_d_sem_opt(), self._get_d_tex_opt()

        lr, ref, hr = self.generator.get_input(batch)
        bsz = lr.shape[0]

        # ref NaN 检查
        if torch.isnan(ref).any() or torch.isinf(ref).any():
            self._nan_d_count += 1
            self._consecutive_nan_d += 1
            self._check_early_stop(is_g_step=False)
            logger.warning(
                "[D step] ref NaN/Inf (#%d), batch=%d", self._nan_d_count, batch_idx
            )
            if d_sem_opt is not None:
                d_sem_opt.zero_grad(set_to_none=True)
                self._d_sem_accum_count = 0
            if d_tex_opt is not None:
                d_tex_opt.zero_grad(set_to_none=True)
                self._d_tex_accum_count = 0
            self._freeze_discriminator()
            self._gd_phase = 0
            return None
        self._consecutive_nan_d = 0

        # ═══════════════════════════════════════════════════════
        # 生成 fake / real（全部 no_grad，D 不需要 G 的梯度）
        # ═══════════════════════════════════════════════════════
        conf_dtex = None
        try:
            with torch.no_grad():
                with torch.amp.autocast(
                    self.device.type, enabled=self.use_amp, dtype=torch.bfloat16
                ):
                    sr_latent = self._get_sr_latent_precomputed(lr, ref)
                    t = self._sample_train_timesteps(
                        bsz,
                        lr.device,
                        t_min=self.aux_t_min,
                        t_max=self.aux_t_max,
                    )
                    noise = torch.randn_like(sr_latent)
                    pred_hr_pixel = self._no_adapter_pred_x0(hr, sr_latent, t, noise)
                    if self.dtex_conf_weight:
                        pred_sr_pixel, conf_dtex = self._adapter_pred_x0(
                            lr, ref, sr_latent, t, noise, return_conf=True
                        )
                    else:
                        pred_sr_pixel = self._adapter_pred_x0(
                            lr, ref, sr_latent, t, noise
                        )
                    real, fake = (
                        pred_hr_pixel.detach().float(),
                        pred_sr_pixel.detach().float(),
                    )
        except (RuntimeError, TypeError, AttributeError) as e:
            err_msg = str(e)
            is_cuda_error = isinstance(e, RuntimeError) and (
                "CUDA" in err_msg or "cuda" in err_msg or "CUBLAS" in err_msg
            )
            if is_cuda_error:
                logger.warning(
                    "[D step] CUDA 异常 (batch=%d): %s，跳过",
                    batch_idx,
                    e,
                    exc_info=True,
                )
                try:
                    torch.cuda.empty_cache()
                except RuntimeError:
                    pass
                if d_sem_opt is not None:
                    d_sem_opt.zero_grad(set_to_none=True)
                if d_tex_opt is not None:
                    d_tex_opt.zero_grad(set_to_none=True)
                self._d_sem_accum_count = 0
                self._d_tex_accum_count = 0
                self._freeze_discriminator()
                self._gd_phase = 0
                return None
            raise

        # ═══════════════════════════════════════════════════════
        # 随机裁剪 patch（与 G step 保持一致，防止分布偏移）
        # ═══════════════════════════════════════════════════════
        crop_i = crop_j = None
        if self.gan_crop_size > 0 and fake.shape[-1] > self.gan_crop_size:
            crop_i = torch.randint(
                0, fake.shape[-2] - self.gan_crop_size + 1, (1,)
            ).item()
            crop_j = torch.randint(
                0, fake.shape[-1] - self.gan_crop_size + 1, (1,)
            ).item()
            fake_crop = fake[
                :,
                :,
                crop_i : crop_i + self.gan_crop_size,
                crop_j : crop_j + self.gan_crop_size,
            ]
            real_crop = real[
                :,
                :,
                crop_i : crop_i + self.gan_crop_size,
                crop_j : crop_j + self.gan_crop_size,
            ]
            # 按比例裁剪 conf_dtex (通常 conf 是 60x60，fake 是 480x480，比例是 1/8)
            if conf_dtex is not None:
                scale_h = conf_dtex.shape[-2] / fake.shape[-2]
                scale_w = conf_dtex.shape[-1] / fake.shape[-1]
                conf_dtex_crop = conf_dtex[
                    :,
                    :,
                    int(crop_i * scale_h) : int(
                        (crop_i + self.gan_crop_size) * scale_h
                    ),
                    int(crop_j * scale_w) : int(
                        (crop_j + self.gan_crop_size) * scale_w
                    ),
                ]
            else:
                conf_dtex_crop = None
        else:
            fake_crop = fake
            real_crop = real
            conf_dtex_crop = conf_dtex

        # NaN 检查 (fake_crop, real_crop)
        for name, tensor in [("fake", fake_crop), ("real", real_crop)]:
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                self._nan_d_count += 1
                self._consecutive_nan_d += 1
                self._check_early_stop(is_g_step=False)
                logger.warning("[D step] %s NaN/Inf (#%d)", name, self._nan_d_count)
                if d_sem_opt is not None:
                    d_sem_opt.zero_grad(set_to_none=True)
                    self._d_sem_accum_count = 0
                if d_tex_opt is not None:
                    d_tex_opt.zero_grad(set_to_none=True)
                    self._d_tex_accum_count = 0
                self._freeze_discriminator()
                self._gd_phase = 0
                return None
        self._consecutive_nan_d = 0

        sem_updated = tex_updated = False

        # ═══════════════════════════════════════════════════════
        # 语义 D
        # ═══════════════════════════════════════════════════════
        if (
            self.lambda_gan_semantic > 0
            and self.discriminator.use_semantic_d
            and d_sem_opt is not None
        ):
            with torch.amp.autocast(self.device.type, enabled=False):
                loss_d_sem = self.discriminator.compute_d_loss(
                    real_crop,
                    fake_crop,
                    ref=None,
                    lambda_semantic=1.0,
                    lambda_texture=0.0,
                )

            if not torch.isnan(loss_d_sem) and not torch.isinf(loss_d_sem):
                (loss_d_sem / self.accumulate_grad_batches).backward()
                self._d_sem_accum_count += 1
                sem_updated = True
                self.log(
                    "train/D_sem", loss_d_sem.detach(), on_step=True, prog_bar=True
                )

                if self._d_sem_accum_count >= self.accumulate_grad_batches:
                    self._monitor_grad_norms(d_sem_opt, "D_sem")
                    self.clip_gradients(
                        d_sem_opt,
                        gradient_clip_val=self.grad_clip_val,
                        gradient_clip_algorithm="norm",
                    )
                    d_sem_opt.step()
                    d_sem_opt.zero_grad(set_to_none=True)
                    self._d_sem_accum_count = 0
            else:
                self._nan_d_count += 1
                logger.warning(
                    "[D step] loss_d_sem NaN/Inf (#%d)，跳过", self._nan_d_count
                )

        # ═══════════════════════════════════════════════════════
        # 纹理 D
        # ═══════════════════════════════════════════════════════
        if (
            self.lambda_gan_texture > 0
            and self.discriminator.use_texture_d
            and d_tex_opt is not None
        ):
            # ── 方案C：Swap Test ──
            # 注意：conf_dtex_crop 不随 ref_for_d 交换。conf 标记的是"fake 的
            # 哪些区域含有 ref 借来的纹理"，这些区域正是与错误 ref 比对时
            # 能暴露不匹配的位置，保持原样即可。
            if self.use_swap_test and bsz > 1:
                n_swap = int(bsz * self.swap_ratio)
                ref_for_d = ref.clone()
                if n_swap > 0:
                    ref_for_d[-n_swap:] = torch.roll(ref[-n_swap:], shifts=1, dims=0)
            else:
                ref_for_d = ref

            # 与 fake_crop/real_crop 使用相同坐标裁剪 ref
            if crop_i is not None:
                ref_for_d_crop = ref_for_d[
                    :,
                    :,
                    crop_i : crop_i + self.gan_crop_size,
                    crop_j : crop_j + self.gan_crop_size,
                ].float()
            else:
                ref_for_d_crop = ref_for_d.float()

            with torch.amp.autocast(self.device.type, enabled=False):
                loss_d_tex = self.discriminator.compute_d_loss(
                    real_crop,
                    fake_crop,
                    ref=ref_for_d_crop,
                    lambda_semantic=0.0,
                    lambda_texture=1.0,
                    tex_weight=conf_dtex_crop,
                )

            if not torch.isnan(loss_d_tex) and not torch.isinf(loss_d_tex):
                (loss_d_tex / self.accumulate_grad_batches).backward()
                self._d_tex_accum_count += 1
                tex_updated = True
                self.log(
                    "train/D_tex", loss_d_tex.detach(), on_step=True, prog_bar=True
                )

                if self._d_tex_accum_count >= self.accumulate_grad_batches:
                    self._monitor_grad_norms(d_tex_opt, "D_tex")
                    self.clip_gradients(
                        d_tex_opt,
                        gradient_clip_val=self.grad_clip_val,
                        gradient_clip_algorithm="norm",
                    )
                    d_tex_opt.step()
                    d_tex_opt.zero_grad(set_to_none=True)
                    self._d_tex_accum_count = 0
            else:
                self._nan_d_count += 1
                logger.warning(
                    "[D step] loss_d_tex NaN/Inf (#%d)，跳过", self._nan_d_count
                )

        # 未参与更新的优化器清零
        if d_sem_opt is not None and not sem_updated:
            d_sem_opt.zero_grad(set_to_none=True)
            self._d_sem_accum_count = 0
        if d_tex_opt is not None and not tex_updated:
            d_tex_opt.zero_grad(set_to_none=True)
            self._d_tex_accum_count = 0

        # 判断是否切回 G phase
        d_sem_done = (
            not self.discriminator.use_semantic_d or self._d_sem_accum_count == 0
        )
        d_tex_done = (
            not self.discriminator.use_texture_d or self._d_tex_accum_count == 0
        )

        if d_sem_done and d_tex_done:
            self._freeze_discriminator()
            self._gd_phase = 0
            self._g_steps_since_d = 0

        return None

    # ═══════════════════════════════════════════════════════
    #  验证 / 推理
    # ═══════════════════════════════════════════════════════

    def validation_step(self, batch, batch_idx):
        import traceback

        try:
            lr, ref, hr = self.generator.get_input(batch)
            loss_diff, _ = self.generator.p_losses(lr, ref, hr)

            self.log(
                "val/loss_diff", loss_diff, on_step=False, on_epoch=True, prog_bar=True
            )
            self.log(
                "val_loss_diff", loss_diff, on_step=False, on_epoch=True, prog_bar=True
            )

            with torch.no_grad():
                val_results = self.generator.log_images(
                    batch,
                    steps=self.sample_steps,
                    sr_model=self.sr_model,
                    t_start=self.t_start,
                    guidance_scale=self.guidance_scale,
                    t_stop=self.t_stop,
                    val_seed=42,
                )

            if self.iqa is not None:
                sr_batch, hq_batch = val_results["samples"], val_results["hq"]
                agg = {}
                iqa_failures = 0
                for i in range(len(sr_batch)):
                    try:
                        m = self.iqa.evaluate_single(
                            sr_batch[i].float().cpu().numpy(),
                            hq_batch[i].float().cpu().numpy(),
                        )
                        for k, v in m.items():
                            agg[k] = agg.get(k, 0.0) + v
                    except Exception as e:
                        iqa_failures += 1
                        logger.warning("IQA sample %d 失败: %s", i, e)
                n = len(sr_batch) - iqa_failures
                if n > 0:
                    for k, v in agg.items():
                        self.log(f"val/{k}", v / n, on_epoch=True, prog_bar=True)
                        self.log(f"val_{k}", v / n, on_epoch=True)

                    # ★ 用训练同款 VGG-LPIPS 覆盖 val/lpips
                    # IQA 内置的是 AlexNet LPIPS，与训练 loss（VGG）不一致；
                    # 这里直接用 self.net_lpips 计算，确保验证指标与优化目标对齐。
                    sr_t = torch.stack([s for s in sr_batch]).to(self.device)
                    hq_t = torch.stack([h for h in hq_batch]).to(self.device)
                    # val_results 值域 [0, 1] → LPIPS 期望 [-1, 1]
                    sr_t = sr_t * 2 - 1
                    hq_t = hq_t * 2 - 1
                    with torch.no_grad():
                        val_lpips_vgg = self._get_lpips()(sr_t, hq_t).mean()
                    self.log("val/lpips", val_lpips_vgg, on_epoch=True, prog_bar=True)
                    self.log("val_lpips", val_lpips_vgg, on_epoch=True)
                else:
                    logger.error(
                        "IQA 全部失败 (%d/%d)，本 epoch 无有效指标",
                        iqa_failures,
                        len(sr_batch),
                    )

            if batch_idx < 5 and self.logger is not None:
                self._save_validation_images(val_results, lr, ref, hr)

            del val_results, lr, ref, hr
            return loss_diff

        except torch.cuda.OutOfMemoryError:
            logger.warning("validation_step OOM (batch=%d)，跳过", batch_idx)
            torch.cuda.empty_cache()
            return None
        except Exception as e:
            logger.error(
                "validation_step 异常 (batch=%d): %s\n%s",
                batch_idx,
                e,
                traceback.format_exc(),
            )
            raise  # 非 OOM 异常不吞掉，让训练者知道

    def _save_validation_images(self, val_results, lr, ref, hr):
        if self.logger is None:
            return

        save_dir = os.path.join(self.logger.log_dir, "validation_tmp")
        os.makedirs(save_dir, exist_ok=True)

        with torch.no_grad():
            with torch.amp.autocast(self.device.type, enabled=False):
                sr_prior = (
                    self.sr_model(lr.float(), ref.float())
                    if self.sr_model is not None
                    else None
                )
                if sr_prior is not None:
                    sr_prior = torch.nan_to_num(
                        sr_prior, nan=0.0, posinf=1.0, neginf=-1.0
                    ).clamp(-1.0, 1.0)

        images_to_concat = []
        for image_key in ("lq", "ref", "hq", "samples"):
            if image_key not in val_results:
                continue
            img = val_results[image_key][0]
            pil_img = Image.fromarray(
                (img.float().detach().cpu().permute(1, 2, 0).numpy() * 255)
                .clip(0, 255)
                .astype(np.uint8)
            )
            if image_key == "lq":
                target_size = val_results["samples"].shape[-2:]
                pil_img = pil_img.resize(
                    (target_size[1], target_size[0]), Image.NEAREST
                )
            images_to_concat.append(pil_img)

        if sr_prior is not None:
            sr_img = (
                ((sr_prior.float() + 1.0) / 2.0)[0]
                .detach()
                .cpu()
                .permute(1, 2, 0)
                .numpy()
            )
            sr_img = (
                (np.nan_to_num(sr_img, nan=0.0, posinf=1.0, neginf=0.0) * 255)
                .clip(0, 255)
                .astype(np.uint8)
            )
            images_to_concat.append(Image.fromarray(sr_img))

        if images_to_concat:
            total_w = sum(im.width for im in images_to_concat)
            max_h = max(im.height for im in images_to_concat)
            combined = Image.new("RGB", (total_w, max_h))
            x_offset = 0
            for im in images_to_concat:
                combined.paste(im, (x_offset, 0))
                x_offset += im.width
            combined.save(os.path.join(save_dir, f"step_{self.global_step}.png"))

    # ═══════════════════════════════════════════════════════
    #  Epoch 钩子
    # ═══════════════════════════════════════════════════════

    def on_train_epoch_start(self):
        # 强制冻结模块保持 eval（防止 Lightning 递归 .train() 打开 DropPath）
        if self.sr_model is not None and self.sr_fixed:
            self.sr_model.eval()
        self.generator.freeze_eval_modules()

    def on_validation_epoch_start(self):
        self._freeze_discriminator()
        if self.logger is not None:
            save_dir = os.path.join(self.logger.log_dir, "validation_tmp")
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
                logger.info("已清理 validation_tmp 目录")

    def on_validation_epoch_end(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_train_start(self):
        if self.sr_model is not None:
            self.sr_model.to(self.device)
        self._override_lr_on_resume()
