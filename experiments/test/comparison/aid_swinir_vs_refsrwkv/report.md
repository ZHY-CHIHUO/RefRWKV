# AID zero-shot paired comparison

Same AID/test sample IDs for all methods (n=500, seed=42, x4). Metrics are means of per-image values.

| Method | N | PSNR (dB) | SSIM |
| --- | ---: | ---: | ---: |
| RefSRWKV-SR | 500 | 27.8408 | 0.707355 |
| SwinIR | 500 | 27.7805 | 0.707377 |
| Bicubic | 500 | 27.7645 | 0.704618 |

## Paired differences

`Improved` is the fraction with delta > 0. p is a two-sided paired t-test.

| Comparison | Metric | Mean delta | Median delta | Improved | p-value |
| --- | --- | ---: | ---: | ---: | ---: |
| RefSRWKV-SR - SwinIR | PSNR | 0.060268 | -0.015222 | 36.4% (182/500) | 0.0563466 |
| RefSRWKV-SR - SwinIR | SSIM | -0.000022 | -0.000944 | 29.2% (146/500) | 0.958421 |
| RefSRWKV-SR - Bicubic | PSNR | 0.076266 | 0.117769 | 64.4% (322/500) | 1.41236e-06 |
| RefSRWKV-SR - Bicubic | SSIM | 0.002737 | 0.003095 | 60.0% (300/500) | 0.00051665 |
| SwinIR - Bicubic | PSNR | 0.015999 | 0.121425 | 65.4% (327/500) | 0.66994 |
| SwinIR - Bicubic | SSIM | 0.002759 | 0.004655 | 63.2% (316/500) | 0.00562604 |

Artifacts: `summary.json`, `per_image.csv`, `delta_psnr_histogram.png`.
