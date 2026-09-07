# 对比模型环境

主项目环境是 `rwkv7`。仓库内的十个比较基线均有不依赖额外编译扩展的
`native_compatibility` 实现，因此可以先在该环境中用统一的 HRMS-SCD x4
配置训练和评测。`requirements.txt` 不引入旧版 MMCV、DCNv2 或 MambaSSM，
以免破坏 RefRWKV 的主环境。

下列 YAML 是“官方实现桥接”用的独立环境定义，不会在安装主项目时自动
创建。使用前请先阅读 [`../docs/models/baselines.md`](../docs/models/baselines.md)
中关于官方实现与 native 版本不可混用的说明。

```bash
conda env create -f environments/mambairv2-official.yaml
conda env create -f environments/ttsr-official.yaml
conda env create -f environments/masa-sr-official.yaml
conda env create -f environments/datsr-legacy.yaml
```

创建后使用简洁的环境名激活，例如 `conda activate mambairv2`、
`conda activate masa-sr` 或 `conda activate ttsr`。

这些文件故意不把整个本仓库复制到独立环境中：官方仓库应单独 clone，使用
其原始训练脚本、数据预处理和 checkpoint 格式。若要把官方结果与本项目汇总，
请以相同 HRMS-SCD split、x4、验证频率和 50,000 optimizer steps 重新训练，
并在结果表中标记 `official bridge`。

| 文件 | 环境名 | 用途 | 风险 |
| --- | --- | --- | --- |
| `mambairv2-official.yaml` | `mambairv2` | MambaIR/MambaIRv2 + MambaSSM | `mamba-ssm` / `causal-conv1d` 是 CUDA 扩展，需与 PyTorch 和 NVCC 匹配。 |
| `ttsr-official.yaml` | `ttsr` | 官方 TTSR | 需要旧 torchvision 兼容的 VGG19 特征权重。 |
| `masa-sr-official.yaml` | `masa-sr` | 官方 MASA-SR | 两阶段（rec 后 GAN）训练；统一 L1 比较仅运行其 rec 路径。 |
| `datsr-legacy.yaml` | `datsr-legacy` | 官方 DATSR | 官方说明锁定 PyTorch 1.7.1、CUDA 10.1、MMCV 0.4.4 / DCNv2；当前 CUDA 12.8 / RTX 5060 Ti 不保证可编译。 |

不要在 `rwkv7` 中安装 `mmcv==0.4.4`、DCNv2 或旧版 PyTorch。对于 DATSR，
推荐先用仓库内的 native compatibility 版本；只有需要逐 checkpoint 对齐原论文
时才在兼容的旧机器/容器中使用 legacy 环境。
