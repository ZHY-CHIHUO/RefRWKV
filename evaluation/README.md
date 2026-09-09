# `evaluation/`

这里是统一推理和指标写盘层。

| 文件 | 内容 |
|---|---|
| `runner.py` | 根据 checkpoint 配置构造 loader/model，运行一个 split，保存预测图和 `metrics.json`；1/3 通道写 PNG，其他通道写 float32 TIFF/NPY。 |
| `eval_pyiqa.py` | 可选的 PyIQA 感知指标。 |
| `eval_sewar.py` | Wuhan 的 RMSE、UIQI、PSNR、SAM、ERGAS 等指标。 |

测试入口是 `scripts/test/run.py`。每个 split 的 `metrics.json` 同时保存平均值和逐图值；`scripts/compare_refsr.py` 根据 `sample_ids` 配对多个模型，计算 ΔPSNR、改善比例和配对 t 检验。
