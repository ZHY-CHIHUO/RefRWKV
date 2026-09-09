# `metrics/`

这里保存与任务相关的指标实现。基础 PSNR/SSIM 由评测流程调用，Wuhan 专用指标在 `wuhan.py` 中实现：RMSE、UIQI、PSNR、SAM（弧度/角度）和 ERGAS。

指标的选择由 `configs/test/test.yaml` 的 `test.metrics` 控制；结果写入每个 split 的 `metrics.json`。
