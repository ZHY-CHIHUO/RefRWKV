"""
Callbacks: 训练过程辅助回调函数 / Training Process Auxiliary Callbacks

提供模型检查点、早停和训练可视化功能。
Provides model checkpointing, early stopping, and training visualization.
"""

from typing import Dict, Any
import os

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from .mixins import ImageLoggerMixin


__all__ = [
    "ModelCheckpoint",
    "EarlyStopping",
    "ImageLogger"
]

# ============================================================================
# ImageLogger: 训练过程中定期保存模型输出图片 / Save model output images
# ============================================================================

class ImageLogger(Callback):
    """
    训练可视化回调: 每隔一定步数用当前模型跑几张图并保存到磁盘，
    让你在训练过程中就能肉眼观察效果，不需要等训练完。
    Training visualization callback: every N steps, run inference with the
    current model on a batch and save the outputs to disk so you can visually
    monitor progress without waiting for training to finish.

    用法 / Usage:
        >>> image_logger = ImageLogger(
        ...     log_every_n_steps=2000,    # 每隔多少步保存一次 / save every N steps
        ...     max_images_each_step=4,    # 每次最多保存几张 / max images per save
        ... )
        >>> trainer = pl.Trainer(callbacks=[image_logger])

    要求 / Requirements:
        模型必须混入 ImageLoggerMixin 并实现 log_images(batch) 方法。
        Model must mix in ImageLoggerMixin and implement log_images(batch).
    """

    def __init__(
        self,
        log_every_n_steps: int = 2000,
        max_images_each_step: int = 4,
        log_images_kwargs: Dict[str, Any] = None
    ) -> "ImageLogger":
        """
        Args:
            log_every_n_steps (int):  每隔多少训练步执行一次图片保存
                                      Log images every N training steps
            max_images_each_step (int): 每类图片最多保存几张（拼成 grid）
                                        Max images per category (stitched into grid)
            log_images_kwargs (Dict): 传给 pl_module.log_images() 的额外参数
                                      Extra kwargs passed to pl_module.log_images()
        """
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.max_images_each_step = max_images_each_step
        self.log_images_kwargs = log_images_kwargs or dict()

    # ---- 训练开始时检查模型是否支持图片日志 / Validate on training start ----
    def on_fit_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """
        训练开始时断言模型实现了 ImageLoggerMixin 接口。
        At training start, assert the model implements ImageLoggerMixin.
        """
        assert isinstance(pl_module, ImageLoggerMixin), (
            "模型必须混入 ImageLoggerMixin 并实现 log_images() 方法。"
            "Model must mix in ImageLoggerMixin and implement log_images()."
        )

    # ---- 每个训练 batch 结束后执行 / After each training batch ----
    @rank_zero_only  # 只在主进程执行（多卡训练时避免重复保存）/ Only on rank 0
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: STEP_OUTPUT,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int
    ) -> None:
        """
        在每个训练 batch 结束时，如果满足步数条件，则冻结模型、进行无梯度推理、
        保存输出图片，最后解冻模型继续训练。
        At the end of each training batch, if the step condition is met, freeze
        the model, run no-grad inference, save output images, then unfreeze.
        """
        # 检查是否到了记录图片的时间 / Check if it is time to log images
        if pl_module.global_step % self.log_every_n_steps == 0:
            is_train = pl_module.training

            # ---- 冻结模型: 推理时不需要梯度 / Freeze model for inference ----
            if is_train:
                pl_module.freeze()

            with torch.no_grad():
                # 调用模型实现的 log_images 方法获取输出图片字典
                # Call the model's log_images method to get output image dict
                images: Dict[str, torch.Tensor] = pl_module.log_images(
                    batch, **self.log_images_kwargs
                )

            # ---- 保存图片到磁盘 / Save images to disk ----
            save_dir = os.path.join(
                pl_module.logger.save_dir, "image_log", "train"
            )
            os.makedirs(save_dir, exist_ok=True)

            for image_key in images:
                image = images[image_key].detach().cpu()

                # hq 图片值域为 [-1, 1]，转回 [0, 1]
                # hq images are in [-1, 1], convert back to [0, 1]
                if image_key == "hq":
                    image = (image + 1.0) / 2.0

                # 取最多 max_images_each_step 张图拼成网格
                # Take at most max_images_each_step images and stitch into grid
                N = min(self.max_images_each_step, len(image))
                grid = torchvision.utils.make_grid(image[:N], nrow=4)

                # chw → hwc 格式转换，用于 PIL 保存
                # Convert from chw to hwc format for PIL saving
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1).numpy()
                grid = (grid * 255).clip(0, 255).astype(np.uint8)

                # 命名规则: {类别}_step-{全局步}_e-{epoch}_b-{batch}.png
                # Naming: {category}_step-{global_step}_e-{epoch}_b-{batch}.png
                filename = "{}_step-{:06}_e-{:06}_b-{:06}.png".format(
                    image_key,
                    pl_module.global_step,
                    pl_module.current_epoch,
                    batch_idx
                )
                path = os.path.join(save_dir, filename)
                Image.fromarray(grid).save(path)

            # ---- 解冻模型: 恢复训练 / Unfreeze model: resume training ----
            if is_train:
                pl_module.unfreeze()
