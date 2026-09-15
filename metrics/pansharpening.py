"""Official pansharpening indices for reduced- and full-resolution tests.

Reduced-resolution (with GT): Q2n, SAM (degrees), ERGAS with ratio=4.
Full-resolution (no GT): D_lambda, D_s, QNR from the Vivone toolbox 1.0
(``flag_orig_paper=0``), block size 32, p=q=alpha=beta=1.

Q2n follows the NumPy port of Garzelli/Nencini onions-quality, but stays in
float64.  The toolbox ``astype(int16)``/``uint16`` path is not used: this
repository stores reflectance in ``[0, 1]``, and integer casts would zero it.
"""

from __future__ import annotations

from math import ceil, log2
import numpy as np
import torch

from .wuhan import wuhan_metric_tensors


def _as_nchw(value: torch.Tensor | np.ndarray) -> torch.Tensor:
    if torch.is_tensor(value):
        tensor = value.detach()
    else:
        tensor = torch.as_tensor(value)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError(f"expected NCHW or CHW, got {tuple(tensor.shape)}")
    return tensor.float()


def _as_hwc(value: torch.Tensor | np.ndarray) -> np.ndarray:
    tensor = _as_nchw(value)
    if tensor.shape[0] != 1:
        raise ValueError("pansharpening helpers take one image at a time")
    return tensor[0].permute(1, 2, 0).contiguous().cpu().numpy().astype(np.float64)


def _as_hw(value: torch.Tensor | np.ndarray) -> np.ndarray:
    array = _as_hwc(value)
    if array.shape[-1] == 1:
        return array[..., 0]
    if array.ndim == 2:
        return array
    raise ValueError(f"expected a single-channel image, got {array.shape}")


def _matlab_cubic(x: np.ndarray) -> np.ndarray:
    """MATLAB ``imresize`` bicubic kernel (Keys cubic with ``a = -0.5``)."""
    absx = np.abs(np.asarray(x, dtype=np.float64))
    absx2 = absx * absx
    absx3 = absx2 * absx
    return (
        (1.5 * absx3 - 2.5 * absx2 + 1.0) * (absx <= 1.0)
        + (-0.5 * absx3 + 2.5 * absx2 - 4.0 * absx + 2.0)
        * ((absx > 1.0) & (absx <= 2.0))
    )


def _matlab_resize_contributions(
    in_length: int, out_length: int, scale: float, kernel_width: float = 4.0
) -> tuple[np.ndarray, np.ndarray]:
    """Antialiased 1-D weights used by MATLAB ``imresize``."""
    if scale < 1.0:
        def kernel(x: np.ndarray) -> np.ndarray:
            return scale * _matlab_cubic(scale * x)

        width = kernel_width / scale
    else:
        kernel = _matlab_cubic
        width = kernel_width
    positions = np.arange(1, out_length + 1, dtype=np.float64)
    u = positions / scale + 0.5 * (1.0 - 1.0 / scale)
    left = np.floor(u - width / 2.0)
    taps = int(ceil(width)) + 2
    indices = left[:, None] + np.arange(taps, dtype=np.float64)[None, :] - 1.0
    weights = kernel(u[:, None] - indices - 1.0)
    weight_sum = weights.sum(axis=1, keepdims=True)
    weight_sum[weight_sum == 0.0] = 1.0
    weights = weights / weight_sum
    mirror = np.concatenate(
        [np.arange(in_length), np.arange(in_length - 1, -1, -1)]
    )
    indices = mirror[np.mod(indices.astype(np.int64), mirror.size)]
    keep = np.any(np.abs(weights) > 1.0e-16, axis=0)
    return weights[:, keep], indices[:, keep]


def matlab_imresize(image: np.ndarray, scale: float) -> np.ndarray:
    """MATLAB ``imresize(..., method='bicubic')`` including antialiasing.

    Toolbox 1.0 ``D_s`` downsamples PAN with this operator *before* the
    23-tap interpolator.  PyTorch/OpenCV bicubic uses ``a = -0.75`` and no
    antialias, which shifts Q_low by ~0.06 and inflates Ds from ~0.027 to ~0.07.
    """
    if scale <= 0.0 or not np.isfinite(scale):
        raise ValueError(f"scale must be a positive finite number, got {scale!r}")
    img = np.asarray(image, dtype=np.float64)
    squeeze = False
    if img.ndim == 2:
        img = img[..., None]
        squeeze = True
    if img.ndim != 3:
        raise ValueError(f"matlab_imresize expects HW or HWC, got {img.shape}")
    height, width = img.shape[:2]
    out_h = int(ceil(scale * height))
    out_w = int(ceil(scale * width))
    if out_h < 1 or out_w < 1:
        raise ValueError(f"matlab_imresize produced empty output {(out_h, out_w)}")
    weights_h, index_h = _matlab_resize_contributions(height, out_h, out_h / height)
    weights_w, index_w = _matlab_resize_contributions(width, out_w, out_w / width)
    resized = np.einsum("op,opwc->owc", weights_h, img[index_h])
    resized = np.einsum("oq,hoqc->hoc", weights_w, resized[:, index_w])
    if squeeze:
        return resized[..., 0]
    return resized


def _cdf23_kernel() -> np.ndarray:
    cdf23 = np.asarray(
        [
            0.5,
            0.305334091185,
            0.0,
            -0.072698593239,
            0.0,
            0.021809577942,
            0.0,
            -0.005192756653,
            0.0,
            0.000807762146,
            0.0,
            -0.000060081482,
        ],
        dtype=np.float64,
    )
    cdf23 = cdf23 * 2.0
    return np.concatenate([np.flip(cdf23[1:]), cdf23])


def interp23tap(image: np.ndarray, ratio: int) -> np.ndarray:
    """23-tap polynomial upsampler used by the QNR toolbox."""
    import scipy.ndimage.filters as nd_filters

    if ratio < 1 or (ratio & (ratio - 1)) != 0:
        raise ValueError(f"interp23tap ratio must be a power of two, got {ratio}")
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 2:
        img = img[..., None]
    if img.ndim != 3:
        raise ValueError(f"interp23tap expects HWC, got {img.shape}")
    kernel = np.expand_dims(_cdf23_kernel(), axis=-1)
    height, width, bands = img.shape
    for step in range(int(log2(ratio))):
        canvas = np.zeros(((2 ** (step + 1)) * height, (2 ** (step + 1)) * width, bands), dtype=np.float64)
        if step == 0:
            canvas[1::2, 1::2, :] = img
        else:
            canvas[::2, ::2, :] = img
        for band in range(bands):
            temp = nd_filters.convolve(np.transpose(canvas[:, :, band]), kernel, mode="wrap")
            canvas[:, :, band] = nd_filters.convolve(np.transpose(temp), kernel, mode="wrap")
        img = canvas
    return img


def _normalize_block(image: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(np.mean(image))
    std = float(np.std(image, ddof=1))
    if std == 0.0:
        std = 1.0e-10
    return (image - mean) / std + 1.0, mean, std


def _cayley_1d(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    size = int(left.shape[0])
    if size <= 1:
        return left * right
    half = size // 2
    a, b = left[:half], left[half:]
    c, d = right[:half], right[half:]
    sign = np.ones_like(b)
    sign[1:] = -1
    b = b * sign
    d = d * sign
    if size == 2:
        return np.concatenate([(a * c) - (d * b), (a * d) + (c * b)])
    ris1 = _cayley_1d(a, c)
    ris2 = _cayley_1d(d, b * sign)
    ris3 = _cayley_1d(a * sign, d)
    ris4 = _cayley_1d(c, b)
    return np.concatenate([ris1 - ris2, ris3 + ris4])


def _cayley_2d(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    depth = int(left.shape[-1])
    if depth <= 1:
        return left * right
    half = depth // 2
    a, b = left[..., :half], left[..., half:]
    c, d = right[..., :half], right[..., half:]
    b = np.concatenate([b[..., :1], -b[..., 1:]], axis=-1)
    d = np.concatenate([d[..., :1], -d[..., 1:]], axis=-1)
    if depth == 2:
        return np.concatenate([(a * c) - (d * b), (a * d) + (c * b)], axis=-1)
    b_neg = np.concatenate([b[..., :1], -b[..., 1:]], axis=-1)
    a_neg = np.concatenate([a[..., :1], -a[..., 1:]], axis=-1)
    ris1 = _cayley_2d(a, c)
    ris2 = _cayley_2d(d, b_neg)
    ris3 = _cayley_2d(a_neg, d)
    ris4 = _cayley_2d(c, b)
    return np.concatenate([ris1 - ris2, ris3 + ris4], axis=-1)


def _q_index_block(reference: np.ndarray, fused: np.ndarray, size: int) -> np.ndarray:
    im1 = np.array(reference, dtype=np.float64, copy=True)
    im2 = np.array(fused, dtype=np.float64, copy=True)
    im2 = np.concatenate([im2[..., :1], -im2[..., 1:]], axis=-1)
    depth = im1.shape[-1]
    for index in range(depth):
        im1[..., index], mean, std = _normalize_block(im1[..., index])
        if mean == 0.0:
            if index == 0:
                im2[..., index] = im2[..., index] - mean + 1.0
            else:
                im2[..., index] = -(-im2[..., index] - mean + 1.0)
        elif index == 0:
            im2[..., index] = ((im2[..., index] - mean) / std) + 1.0
        else:
            im2[..., index] = -(((-im2[..., index] - mean) / std) + 1.0)
    m1 = np.mean(im1, axis=(0, 1))
    m2 = np.mean(im2, axis=(0, 1))
    mod_q1m = np.sqrt(np.sum(m1 ** 2))
    mod_q2m = np.sqrt(np.sum(m2 ** 2))
    mod_q1 = np.sqrt(np.sum(im1 ** 2, axis=-1))
    mod_q2 = np.sqrt(np.sum(im2 ** 2, axis=-1))
    term2 = mod_q1m * mod_q2m
    term4 = mod_q1m ** 2 + mod_q2m ** 2
    temp = (size ** 2) / (size ** 2 - 1)
    term3 = temp * np.mean(mod_q1 ** 2) + temp * np.mean(mod_q2 ** 2) - temp * term4
    mean_bias = 2.0 * term2 / term4
    if term3 == 0.0:
        q = np.zeros(depth, dtype=np.float64)
        q[-1] = mean_bias
        return q
    qu = _cayley_2d(im1, im2)
    qm = _cayley_1d(m1, m2)
    q = (temp * np.mean(qu, axis=(0, 1)) - temp * qm) * mean_bias * (2.0 / term3)
    return np.asarray(q, dtype=np.float64).reshape(-1)


def q2n(
    prediction: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    *,
    block_size: int = 32,
    shift: int = 32,
) -> torch.Tensor:
    """Hypercomplex Q2n / Q8 on one NCHW batch, one scalar per image."""
    pred = _as_nchw(prediction)
    truth = _as_nchw(target)
    if pred.shape != truth.shape:
        raise ValueError(f"Q2n shape mismatch: {tuple(pred.shape)} vs {tuple(truth.shape)}")
    scores = []
    for index in range(pred.shape[0]):
        fused = pred[index].permute(1, 2, 0).cpu().numpy().astype(np.float64)
        labels = truth[index].permute(1, 2, 0).cpu().numpy().astype(np.float64)
        height, width, depth = labels.shape
        step_x = max(1, int(ceil(height / shift)))
        step_y = max(1, int(ceil(width / shift)))
        pad_h = (step_x - 1) * shift + block_size - height
        pad_w = (step_y - 1) * shift + block_size - width
        if pad_h or pad_w:
            labels = np.pad(labels, ((0, max(pad_h, 0)), (0, max(pad_w, 0)), (0, 0)), mode="reflect")
            fused = np.pad(fused, ((0, max(pad_h, 0)), (0, max(pad_w, 0)), (0, 0)), mode="reflect")
        height, width, depth = labels.shape
        if ceil(log2(depth)) - log2(depth) != 0:
            extra = (2 ** int(ceil(log2(depth)))) - depth
            zeros = np.zeros((height, width, extra), dtype=np.float64)
            labels = np.concatenate([labels, zeros], axis=-1)
            fused = np.concatenate([fused, zeros], axis=-1)
            depth = labels.shape[-1]
        values = np.zeros((step_x, step_y, depth), dtype=np.float64)
        for row in range(step_x):
            for col in range(step_y):
                y0, x0 = row * shift, col * shift
                values[row, col] = _q_index_block(
                    labels[y0 : y0 + block_size, x0 : x0 + block_size],
                    fused[y0 : y0 + block_size, x0 : x0 + block_size],
                    block_size,
                )
        scores.append(float(np.mean(np.sqrt(np.sum(values ** 2, axis=-1)))))
    return torch.tensor(scores, dtype=torch.float64)


def _uqi_mean(left: np.ndarray, right: np.ndarray, block: int) -> float:
    height, width = left.shape
    if height % block or width % block:
        raise ValueError(f"UQI block {block} must divide {height}x{width}")
    rows, cols = height // block, width // block
    left_b = left.reshape(rows, block, cols, block).transpose(0, 2, 1, 3).reshape(-1, block * block)
    right_b = right.reshape(rows, block, cols, block).transpose(0, 2, 1, 3).reshape(-1, block * block)
    mean_l = left_b.mean(axis=1)
    mean_r = right_b.mean(axis=1)
    count = block * block
    var_l = ((left_b - mean_l[:, None]) ** 2).sum(axis=1) / (count - 1)
    var_r = ((right_b - mean_r[:, None]) ** 2).sum(axis=1) / (count - 1)
    cov = ((left_b - mean_l[:, None]) * (right_b - mean_r[:, None])).sum(axis=1) / (count - 1)
    numerator = 4.0 * cov * mean_l * mean_r
    denominator = (var_l + var_r) * (mean_l ** 2 + mean_r ** 2)
    quality = np.ones_like(numerator)
    valid = np.abs(denominator) > 1.0e-20
    quality[valid] = numerator[valid] / denominator[valid]
    return float(quality.mean())


def d_lambda(
    fused: torch.Tensor | np.ndarray,
    ms_up: torch.Tensor | np.ndarray,
    *,
    block_size: int = 32,
    exponent: float = 1.0,
) -> torch.Tensor:
    """Spectral distortion index (toolbox 1.0, compared at PAN scale)."""
    fused_b = _as_nchw(fused)
    ms_b = _as_nchw(ms_up)
    scores = []
    for index in range(fused_b.shape[0]):
        fused_hwc = fused_b[index].permute(1, 2, 0).cpu().numpy().astype(np.float64)
        ms_hwc = ms_b[index].permute(1, 2, 0).cpu().numpy().astype(np.float64)
        bands = fused_hwc.shape[-1]
        total = 0.0
        pairs = 0
        for left in range(bands - 1):
            for right in range(left + 1, bands):
                expected = _uqi_mean(ms_hwc[..., left], ms_hwc[..., right], block_size)
                actual = _uqi_mean(fused_hwc[..., left], fused_hwc[..., right], block_size)
                total += abs(actual - expected) ** exponent
                pairs += 1
        scores.append((total / pairs) ** (1.0 / exponent))
    return torch.tensor(scores, dtype=torch.float64)


def d_s(
    fused: torch.Tensor | np.ndarray,
    ms_up: torch.Tensor | np.ndarray,
    pan: torch.Tensor | np.ndarray,
    *,
    ratio: int = 4,
    block_size: int = 32,
    exponent: float = 1.0,
) -> torch.Tensor:
    """Spatial distortion index (toolbox 1.0: interp23tap of MATLAB-imresize PAN)."""
    fused_b = _as_nchw(fused)
    ms_b = _as_nchw(ms_up)
    pan_b = _as_nchw(pan)
    scores = []
    for index in range(fused_b.shape[0]):
        fused_hwc = fused_b[index].permute(1, 2, 0).cpu().numpy().astype(np.float64)
        ms_hwc = ms_b[index].permute(1, 2, 0).cpu().numpy().astype(np.float64)
        pan_hw = pan_b[index, 0].cpu().numpy().astype(np.float64)
        pan_lr = matlab_imresize(pan_hw, 1.0 / float(ratio))
        pan_filt = interp23tap(pan_lr, ratio)[..., 0]
        if pan_filt.shape != pan_hw.shape:
            pan_filt = pan_filt[: pan_hw.shape[0], : pan_hw.shape[1]]
        bands = fused_hwc.shape[-1]
        total = 0.0
        for band in range(bands):
            high = _uqi_mean(fused_hwc[..., band], pan_hw, block_size)
            low = _uqi_mean(ms_hwc[..., band], pan_filt, block_size)
            total += abs(high - low) ** exponent
        scores.append((total / bands) ** (1.0 / exponent))
    return torch.tensor(scores, dtype=torch.float64)


def qnr(
    fused: torch.Tensor | np.ndarray,
    ms_up: torch.Tensor | np.ndarray,
    pan: torch.Tensor | np.ndarray,
    *,
    ratio: int = 4,
    block_size: int = 32,
    p: float = 1.0,
    q: float = 1.0,
    alpha: float = 1.0,
    beta: float = 1.0,
) -> dict[str, torch.Tensor]:
    spectral = d_lambda(fused, ms_up, block_size=block_size, exponent=p)
    spatial = d_s(fused, ms_up, pan, ratio=ratio, block_size=block_size, exponent=q)
    score = ((1.0 - spectral) ** alpha) * ((1.0 - spatial) ** beta)
    return {"d_lambda": spectral, "d_s": spatial, "qnr": score}


def reduced_resolution_tensors(
    prediction: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    *,
    ratio: float = 4.0,
) -> dict[str, torch.Tensor]:
    """PSNR-band SAM/ERGAS plus Q2n for one reduced-resolution batch."""
    values = wuhan_metric_tensors(
        prediction,
        target,
        resolution_ratio=ratio,
        value_range="zero_one",
    )
    return {
        "sam_deg": values["sam_deg"],
        "sam_rad": values["sam_rad"],
        "ergas": values["ergas"],
        "q2n": q2n(prediction, target),
        "psnr": values["psnr"],
    }


__all__ = [
    "d_lambda",
    "d_s",
    "interp23tap",
    "matlab_imresize",
    "q2n",
    "qnr",
    "reduced_resolution_tensors",
]
