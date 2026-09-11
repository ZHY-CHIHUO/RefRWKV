# `models/`

这里保存 SR、RefSR 和 RefDiffRWKV 的网络实现及 registry/adapter。训练 engine 只依赖统一的模型接口，不直接复制某个模型的训练循环。

## 目录

| 路径 | 内容 |
|---|---|
| `sr/` | 单图 SR registry、SwinIR，以及 Bicubic、EDSR、RCAN、HAT、MambaIRv2 compatibility baseline。 |
| `refsr/` | direct RefSR registry、RefSRWKV、TTSR、MASA-SR、DATSR 和 RefDiffRWKV。 |

SR 模型通常提供 `forward(lr)`；direct RefSR 模型提供 `forward(lr, ref)`。模型输入统一遵守项目的 `[-1, 1]` tensor 契约，Bicubic 是无参数的 `reference_only` 评测基线。

## 模型家族

- `models/sr/swinir/`：项目现有 SwinIR 网络和 adapter。
- `models/sr/baselines.py`、`baseline_adapters.py`：轻量单图对比模型和 registry 适配。
- `models/refsr/refsrwkv/`：参考图超分 RWKV，支持 `paired` 真实 Ref 和 `lr_up` 自参考两种数据模式。
- `models/refsr/baselines.py`、`baseline_adapters.py`：TTSR、MASA-SR、DATSR 的统一 direct RefSR 适配。
- `models/refsr/RefDiffRWKV/`：扩散生成器、SR prior、参考适配器、判别器和采样器。

HRMS-SCD x4 的共同比较协议在 `configs/common/benchmark.yaml`；模型配置在 `configs/models/`，一次实验组合在 `configs/runs/`。官方实现与本仓库的 `native_compatibility` 结果不能混用。

## RefSRWKV 多通道契约

`RefSRWKV` 的 `model.inp_channels` 是 LR/输出通道数，`model.ref_channels` 是参考图通道数，必须满足
`1 <= ref_channels <= inp_channels` 和 `out_channels == inp_channels`。因此同一个网络结构可以表示：

```yaml
model:
  inp_channels: 4   # LR 与输出 HR
  ref_channels: 1   # PAN 引导
  out_channels: 4
```

或者 `inp_channels: 8, ref_channels: 3` 的高光谱/多光谱引导。`data.scale` 仍是该模型实例的固定整数倍率，输入必须满足 `Ref_H/W = LR_H/W * scale`，但 LR 高宽可以任意，网络会在内部按需补齐并裁回原尺寸。通道数不一致时，`color_match: global` 使用每张图的跨通道全局统计匹配，不复制或丢弃光谱带；同通道配置继续使用原来的逐通道统计匹配。

## RefSRWKV 三分支融合

Python 构造函数中的 `fusion_mode` 默认为 `legacy`，与已有 checkpoint 的 `GatedFusion` 完全兼容；
项目配置 `configs/models/refsr/refsrwkv.yaml` 默认使用新的 `spectral_detail`。该模式将参考图先在 HR 网格分解为低频 `Ref_low` 和高频
`Ref_high = Ref - Ref_low`，再通过同一组编码器下采样到 LR/U-Net 的四个尺度。

- LR 主干保留低频和光谱信息，不被参考特征直接替换。
- `LR_low`、`Ref_low` 和绝对差异用于光谱关系建模，产生 `g_spec` 控制的低频残差。
- `LR_low` 作为 query、`Ref_low` 作为 key、`Ref_high` 作为 value 做局部匹配；熵置信度和质量估计形成可靠性。
- 高频只经 `g_detail * reliability` 控制的 FiLM/SFT 残差调制 LR 特征，FiLM 最后一层零初始化，因此新模块初始等价于原 LR/SISR 路径。

该设计保留两个门控：`g_spec` 管低频/光谱参考，`g_detail` 管高频空间细节；二者职责不同，不能合并为一个总门控。`fusion_match.enabled/conf/quality` 仍可用于消融局部匹配、熵置信度和质量门控。

`g_spec` 和 `g_detail` 是独立的模型配置开关，不是“参考信息不足”时的自适应开关，而是用于归因消融：

- `g_spec=true, g_detail=false`：Ref 只提供低频/光谱校正；
- `g_spec=false, g_detail=true`：Ref 只提供高频/空间细节；
- `g_spec=false, g_detail=false`：跳过参考路径，作为 SISR 下界基线。

两条路径关闭时不执行参考颜色匹配、参考金字塔和局部高频匹配，可直接用同一个 paired 测试配置完成三档对比。
