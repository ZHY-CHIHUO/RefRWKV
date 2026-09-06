"""SR task entry point for the unified super-resolution Dataset."""

from __future__ import annotations

from pathlib import Path

from data.dataset import SuperResolutionDataset


class SRPNGDataset(SuperResolutionDataset):
    """SR-oriented HR/LR wrapper around :class:`SuperResolutionDataset`.

    The wrapper fixes the returned fields to ``("lr", "hr")``.
    """

    def __init__(
        self,
        data_dir: str | Path,
        mode: str = "train",
        patch_size: int | None = None,
        scale: int = 4,
        augment: bool = False,
        max_samples: tuple[int | None, int | None, int | None] = (None, None, None),
        sample_seed: int = 42,
        lr_key: str = "lr",
        hr_key: str = "hr",
        **kwargs,
    ):
        kwargs.pop("return_items", None)
        kwargs.pop("return_ref", None)
        kwargs.pop("reference_source", None)
        super().__init__(
            data_dir=data_dir,
            mode=mode,
            patch_size=patch_size,
            scale=scale,
            augment=augment,
            max_samples=max_samples,
            sample_seed=sample_seed,
            lr_key=lr_key,
            hr_key=hr_key,
            return_items=("lr", "hr"),
            reference_source="none",
            **kwargs,
        )


__all__ = ["SRPNGDataset"]
