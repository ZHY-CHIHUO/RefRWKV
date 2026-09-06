"""RefSR task entry point for the unified reference SR Dataset."""

from __future__ import annotations

from pathlib import Path

from data.dataset import SuperResolutionDataset


class RefPNGDataset(SuperResolutionDataset):
    """RefSR-oriented HR/LR/Ref wrapper around :class:`SuperResolutionDataset`.

    The wrapper fixes the returned fields to ``("lr", "hr", "ref")``.
    """

    def __init__(
        self,
        data_dir: str | Path,
        mode: str = "train",
        patch_size: int | None = None,
        scale: int = 10,
        augment: bool = False,
        augment_ref: bool = False,
        ref_aug_strengths: list[float] | None = None,
        ref_aug_probs: list[float] | None = None,
        ref_gray_prob: float = 0.2,
        max_samples: tuple[int | None, int | None, int | None] = (None, None, None),
        sample_seed: int = 42,
        lr_key: str = "lr",
        hr_key: str = "hr",
        ref_key: str = "ref",
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
            augment_ref=augment_ref,
            ref_aug_strengths=ref_aug_strengths,
            ref_aug_probs=ref_aug_probs,
            ref_gray_prob=ref_gray_prob,
            max_samples=max_samples,
            sample_seed=sample_seed,
            lr_key=lr_key,
            hr_key=hr_key,
            ref_key=ref_key,
            return_items=("lr", "hr", "ref"),
            reference_source="stored",
            **kwargs,
        )


__all__ = ["RefPNGDataset"]
