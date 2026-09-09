# `scripts/prepare/`

这里保存数据准备和检查脚本：

- `remote_sensing.py`：整理常规遥感图像为 HR/LR split，并按配置生成 bicubic LR；
- `wuhan.py`：整理 Wuhan temporal-pair TIFF 数据和 split 元数据。

脚本只写入 `data/` 下的本地数据目录；训练/测试入口不会自动下载数据。
