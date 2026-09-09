# `losses/`

这里保存可组合训练损失。`common.py` 提供 L1、SSIM、FFT 等基础项，具体启用和权重由 YAML 的 `loss.*` 控制。

损失只参与训练 engine；测试指标由 `evaluation/` 和 `metrics/` 计算，不能把训练损失直接当作测试指标。
