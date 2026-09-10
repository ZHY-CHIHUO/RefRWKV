# AID/test paired comparison

All rows use the same sample IDs. PSNR/SSIM are means over per-image values.

| Method | N | PSNR (dB) | SSIM | ΔPSNR vs Bicubic (dB) | ΔPSNR vs SwinIR (dB) | ΔPSNR vs RefSRWKV-SR (dB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bicubic | 1493 | 27.5785 | 0.6990 | 0.0000 | -0.0091 | -0.0827 |
| SwinIR | 1493 | 27.5876 | 0.7020 | 0.0091 | 0.0000 | -0.0736 |
| RefSRWKV-SR | 1493 | 27.6612 | 0.7022 | 0.0827 | 0.0736 | 0.0000 |
| TRefSR | 1493 | 27.5862 | 0.6997 | 0.0077 | -0.0014 | -0.0750 |

## Per-image significance

`improved` is the fraction of images with ΔPSNR > 0. The p-value is a two-sided paired t-test over the same image IDs.

| Comparison | Mean Δ (dB) | Median Δ (dB) | Improved | t | p |
| --- | ---: | ---: | ---: | ---: | ---: |
| TRefSR - Bicubic | 0.0077 | 0.0665 | 58.94% | 0.617 | 0.537630 |
| TRefSR - SwinIR | -0.0014 | -0.0491 | 21.37% | -0.076 | 0.939746 |
| TRefSR - RefSRWKV-SR | -0.0750 | -0.0408 | 19.16% | -8.896 | 0.000000 |

## Pairing provenance

- Bicubic: `metrics.sample_ids`
- SwinIR: `metrics.sample_ids`
- RefSRWKV-SR: `metrics.sample_ids`
- TRefSR: `metrics.sample_ids`
