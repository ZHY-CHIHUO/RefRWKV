# test_hard paired comparison

All rows use the same sample IDs. PSNR/SSIM are means over per-image values.

| Method | N | PSNR (dB) | SSIM | ΔPSNR vs Bicubic (dB) | ΔPSNR vs SwinIR (dB) | ΔPSNR vs RefSRWKV-SR (dB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bicubic | 500 | 23.0630 | 0.5929 | 0.0000 | -2.5788 | -2.4325 |
| SwinIR | 500 | 25.6418 | 0.7087 | 2.5788 | 0.0000 | 0.1463 |
| RefSRWKV-SR | 500 | 25.4955 | 0.7032 | 2.4325 | -0.1463 | 0.0000 |
| TRefSR | 500 | 25.6680 | 0.7121 | 2.6050 | 0.0262 | 0.1725 |

## Per-image significance

`improved` is the fraction of images with ΔPSNR > 0. The p-value is a two-sided paired t-test over the same image IDs.

| Comparison | Mean Δ (dB) | Median Δ (dB) | Improved | t | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| TRefSR - Bicubic | 2.6050 | 2.4903 | 100.00% | 81.188 | 0.000000 |
| TRefSR - SwinIR | 0.0262 | 0.0008 | 50.00% | 2.491 | 0.013055 |
| TRefSR - RefSRWKV-SR | 0.1725 | 0.1321 | 83.60% | 17.751 | 0.000000 |

## Pairing provenance

- Bicubic: `metrics.sample_ids`
- SwinIR: `metrics.sample_ids`
- RefSRWKV-SR: `metrics.sample_ids`
- TRefSR: `metrics.sample_ids`
