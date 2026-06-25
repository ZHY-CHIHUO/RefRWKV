"""
ImageLoggerMixin: 图片日志接口 / Image Logging Interface

这是一个 Mixin 接口类，使用 Python 的类型重载 (@overload) 声明子类必须实现
log_images 方法。自身不提供任何实现，只定义契约。
A Mixin interface class that uses @overload to declare the log_images method
signature that subclasses must implement. It provides no implementation itself,
only the contract.
"""

from typing import overload, Any, Dict
import torch


class ImageLoggerMixin:
    """
    图片日志混入接口 / Image Logging Mixin Interface

    任何需要配合 ImageLogger Callback 使用的 PyTorch Lightning 模型
    都必须继承此类并实现 log_images() 方法。
    Any PyTorch Lightning model that works with the ImageLogger Callback
    must inherit this class and implement log_images().

    用法示例 / Example usage:
        >>> class MySuperResolutionModel(pl.LightningModule, ImageLoggerMixin):
        ...     def log_images(self, batch, **kwargs):
        ...         lr, ref, hr = batch
        ...         sr = self(lr, ref)
        ...         return {
        ...             "lr": lr,          # 低分辨率输入 / low-res input
        ...             "ref": ref,        # 参考图像 / reference image
        ...             "sr": sr,          # 模型超分输出 / model SR output
        ...             "hq": hr,          # 高分辨率真值 / high-res ground truth
        ...         }

    hq 值的特殊处理 / Special handling for "hq":
        ImageLogger 会对 key 为 "hq" 的图像自动做 (x + 1) / 2 的映射
        （从 [-1, 1] 转到 [0, 1]），其他 key 则不做此转换。
        ImageLogger automatically maps images with key "hq" from [-1, 1]
        to [0, 1] via (x + 1) / 2. Other keys are left unchanged.
    """

    @overload
    def log_images(
        self,
        batch: Any,
        **kwargs: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """
        从输入 batch 中生成一组可视化图片 / Generate a set of visualization
        images from the input batch.

        这是一个抽象方法签名 (通过 @overload 声明)。子类必须重写此方法
        提供具体实现，否则运行时 ImageLogger 会在 on_fit_start 阶段通过
        assert isinstance 检查抛出错误。
        This is an abstract method signature (declared via @overload).
        Subclasses must override this method with a concrete implementation;
        otherwise ImageLogger will raise an AssertionError during on_fit_start.

        Args:
            batch (Any): 来自 DataLoader 的一个 batch / A batch from DataLoader
            **kwargs:    额外参数（由 ImageLogger 的 log_images_kwargs 传入）
                         Extra arguments passed via ImageLogger's log_images_kwargs

        Returns:
            Dict[str, torch.Tensor]: 键值对映射，key 为图片类别名（如 "lr", "sr"）
                                      value 为形状 (N,C,H,W)、值域 [0,1] 的 RGB 张量。
                                      Key-value mapping: key is image category name
                                      (e.g., "lr", "sr"), value is an RGB tensor of
                                      shape (N,C,H,W) with values in [0,1].
        """
        ...
