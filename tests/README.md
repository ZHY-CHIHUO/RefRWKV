# `tests/`

这里保存轻量回归测试和模型几何冒烟测试。测试不依赖完整训练数据时优先使用合成 tensor；需要 CUDA/WKV 的测试会在名称或命令中明确说明。

| 文件 | 内容 |
|---|---|
| `test_unified_dataset.py` | HR/LR/Ref 目录、倍率和返回字段。 |
| `test_reference_contract.py` | `lr_up` 与 `paired` 参考图契约。 |
| `test_test_config.py` | 独立 test YAML 与 checkpoint `trainer_config` 合并。 |
| `test_refsrwkv_ablation.py` | RefSRWKV 消融配置和 forward。 |
| `test_baseline_registry.py` | SR/RefSR baseline registry 和输出尺寸。 |
| `test_tiling.py` | 大图 tile 推理拼接。 |
| `test_wuhan_dataset.py` | Wuhan 四通道 temporal-pair 数据契约。 |
| `smoke_native_geometry.py` | 原生模型不同倍率的形状和梯度冒烟测试。 |

运行全部 Python 测试：

```bash
pytest -q
```
