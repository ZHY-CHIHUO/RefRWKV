# 参数量 / FLOPs / 推理时间

来源：`experiments/test/refsr/compare_pan_cost.json`。

- FlopCounterMode **会少算** Mamba / WKV 的 CUDA kernel。真正该看的是 **ms/img**。
- 耗时脚本里的 `test_hard` 实际喂进去的仍是 **64×64 MS / 256×256 PAN**，所以所谓 FR 耗时行和 RR 是同一空间尺寸，不是 OrigScale 128/512。
- RDM-PAN 没法在 256 上做公平的 CUDA 整图，因为 Bi-WKV CUDA 在 T=256 会坏；生产路径就是 16/8 分块。

| 模型 | 权重 | 推理 | 划分标签 | 块数 | 参数量 | FLOPs | ms/图 | 峰值显存 MB | 说明 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fusion_mamba | 420.ckpt | 官方整图 | test | 1 | 0.735M | 19.62G | 52.3 | 346.4 | 官方整图 |
| fusion_mamba | 420.ckpt | tile16_ov8 | test | 49 | 0.735M | 60.09G | 2399.4 | 36.0 | 与 RDM-PAN 相同分块：LR 16 / 重叠 8（PAN 64） |
| fusion_mamba | 420.ckpt | 官方切块 | test_hard | 1 | 0.735M | 19.62G | 53.0 | 346.4 | 官方 cut_size=256 PAN（LR tile=64，重叠=0）。耗时脚本输入仍是 64×64 MS / 256 PAN，不是 OrigScale 128/512 |
| fusion_mamba | 420.ckpt | tile16_ov8 | test_hard | 49 | 0.735M | 60.09G | 2480.5 | 36.0 | 与 RDM-PAN 相同分块：LR 16 / 重叠 8（PAN 64）。耗时脚本输入仍是 64×64 MS / 256 PAN，不是 OrigScale 128/512 |
| rdm_pan | last.ckpt | tile16_ov8 | test | 49 | 0.625M | 103.96G | 2064.2 | 24.5 | RDM-PAN 分块 LR 16 / 重叠 8（PAN 64） |
| rdm_pan | last.ckpt | tile16_ov8 | test_hard | 49 | 0.625M | 103.96G | 2036.9 | 24.5 | RDM-PAN 分块 LR 16 / 重叠 8（PAN 64）。耗时脚本输入仍是 64×64 MS / 256 PAN，不是 OrigScale 128/512 |

同一张 256 图：FusionMamba 官方整图大约 **52 ms**；同样 16/8 分块时 FusionMamba 大约 **2.4 s**，RDM-PAN 大约 **2.06 s**。
