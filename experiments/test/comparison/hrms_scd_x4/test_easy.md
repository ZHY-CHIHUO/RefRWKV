# test_easy paired comparison

All rows use the same sample IDs. PSNR/SSIM are means over per-image values.

| Method | N | PSNR (dB) | SSIM | ΔPSNR vs Bicubic (dB) | ΔPSNR vs SwinIR (dB) | ΔPSNR vs RefSRWKV-SR (dB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bicubic | 500 | 23.1638 | 0.5941 | 0.0000 | -2.3659 | -2.2302 |
| SwinIR | 500 | 25.5297 | 0.7035 | 2.3659 | 0.0000 | 0.1357 |
| RefSRWKV-SR | 500 | 25.3940 | 0.6979 | 2.2302 | -0.1357 | 0.0000 |
| TRefSR | 500 | 25.6744 | 0.7128 | 2.5105 | 0.1447 | 0.2804 |

## Per-image significance

`improved` is the fraction of images with ΔPSNR > 0. The p-value is a two-sided paired t-test over the same image IDs.

| Comparison | Mean Δ (dB) | Median Δ (dB) | Improved | t | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| TRefSR - Bicubic | 2.5105 | 2.3393 | 100.00% | 71.850 | 0.000000 |
| TRefSR - SwinIR | 0.1447 | 0.1080 | 70.00% | 11.377 | 0.000000 |
| TRefSR - RefSRWKV-SR | 0.2804 | 0.2352 | 90.00% | 23.360 | 0.000000 |

## Pairing provenance

- Bicubic: `metrics.sample_ids`
- SwinIR: `metrics.sample_ids`
- RefSRWKV-SR: `metrics.sample_ids`
- TRefSR: `metrics.sample_ids`
