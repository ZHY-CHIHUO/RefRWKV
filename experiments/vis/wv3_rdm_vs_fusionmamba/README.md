# WV3 对比：RDM-PAN vs FusionMamba

本机可视化 + 指标汇总（PNG 被 gitignore）。重建命令：

```bash
/home/zhy/miniconda3/envs/rwkv7/bin/python scripts/test/build_wv3_compare_folder.py
```

## 这个文件夹里有什么

- `images/rr/`：20 张降分辨率图。真值、RDM-PAN（`last.ckpt`，tile 16/8）、FusionMamba 官方可视化，以及 `真值 | RDM | FM` 拼图。
- `images/fr/`：20 张全分辨率图。RDM-PAN（`last.ckpt`，tile 16/8）和 FusionMamba 官方可视化（没有真值）。
- `tables/`：RR / FR 主表、耗时表，以及磁盘上**每一次**完整指标跑次。

图片是拷贝现成结果，没有重新推理：

- RDM-PAN：`experiments/test/refsr/rdm_pan_preview/tile16_ov8/{test,test_hard}`（干净 RGB，无红格；`*_grad.png` 不用）。
- FusionMamba：`experiments/vis/fusion_mamba_official/WV3_{rr,fr}/FusionMamba`。
- 真值：同一 `test` 划分的 RDM `*_gt.png`（官方可视化 GT 也是这 20 张 RR）。

- RR 尺寸：RDM / FusionMamba / 真值都是 256×256。
- FR 尺寸：RDM / FusionMamba 都是 512×512。

## 主表（引用这些数字）

降分辨率（`test`，20 张）：

| 模型 | 权重 | 推理方式 | PSNR↑ | Q2n↑ | SAM°↓ | ERGAS↓ |
| --- | --- | --- | --- | --- | --- | --- |
| 论文 FusionMamba | 论文表格 | 官方（论文） | 39.374±2.973 | 0.922±0.084 | 2.843±0.577 | 2.092±0.510 |
| FusionMamba 官方 420 | 420.ckpt | 整图（tile=1024 不切块） | 39.3434±2.8183 | 0.9192±0.0814 | 2.8163±0.5229 | 2.0795±0.4580 |
| FusionMamba 自训 last | last.ckpt | 整图（tile=1024 不切块） | 39.2891±2.8166 | 0.9182±0.0817 | 2.8277±0.5221 | 2.0910±0.4554 |
| RDM-PAN last（= epoch 439） | last.ckpt | LR tile=16，重叠=8 | 39.3994±2.8494 | 0.9193±0.0820 | 2.7821±0.5257 | 2.0580±0.4346 |
| RDM-PAN epoch 419 | epoch=0419-step=127260 | LR tile=16，重叠=8 | 39.3960±2.8502 | 0.9195±0.0817 | 2.7817±0.5264 | 2.0591±0.4357 |
| RDM-PAN epoch 409 | epoch=0409-step=124230 | LR tile=16，重叠=8 | 39.3988±2.8499 | 0.9194±0.0821 | 2.7830±0.5267 | 2.0581±0.4359 |

全分辨率（`test_hard`，20 张；**toolbox Ds**）：

| 模型 | 权重 | 推理方式 | Dλ↓ | Ds↓ | QNR↑ | Ds 协议 |
| --- | --- | --- | --- | --- | --- | --- |
| 论文 FusionMamba | 论文表格 | 官方（论文） | 0.0186±0.0078 | 0.0269±0.0058 | 0.9550±0.0110 | 论文 |
| FusionMamba 官方 420 | 420.ckpt | LR tile=64，重叠=0 | 0.0166±0.0057 | 0.0270±0.0059 | 0.9569±0.0110 | toolbox MATLAB imresize（与论文一致） |
| FusionMamba 自训 last | last.ckpt | LR tile=64，重叠=0 | 0.0189±0.0048 | 0.0339±0.0032 | 0.9479±0.0073 | toolbox MATLAB imresize（与论文一致） |
| RDM-PAN last（= epoch 439） | last.ckpt | LR tile=16，重叠=8 | 0.0249±0.0064 | 0.0375±0.0029 | 0.9385±0.0085 | toolbox MATLAB imresize（与论文一致） |
| RDM-PAN epoch 419 | epoch=0419-step=127260 | LR tile=16，重叠=8 | 0.0245±0.0066 | 0.0381±0.0034 | 0.9384±0.0091 | toolbox MATLAB imresize（与论文一致） |
| RDM-PAN epoch 409 | epoch=0409-step=124230 | LR tile=16，重叠=8 | 0.0265±0.0075 | 0.0395±0.0038 | 0.9350±0.0104 | toolbox MATLAB imresize（与论文一致） |
| RDM-PAN epoch 279（已删） | epoch=0279（不在盘上） | LR tile=16，重叠=8 | 0.0219 | 0.0378 | 0.9412 | 历史 FR 记录，权重已删 |

## 耗时（256 尺寸图）

| 模型 | 权重 | 推理 | 划分标签 | 块数 | 参数量 | FLOPs | ms/图 | 峰值显存 MB | 说明 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fusion_mamba | 420.ckpt | 官方整图 | test | 1 | 0.735M | 19.62G | 52.3 | 346.4 | 官方整图 |
| fusion_mamba | 420.ckpt | tile16_ov8 | test | 49 | 0.735M | 60.09G | 2399.4 | 36.0 | 与 RDM-PAN 相同分块：LR 16 / 重叠 8（PAN 64） |
| fusion_mamba | 420.ckpt | 官方切块 | test_hard | 1 | 0.735M | 19.62G | 53.0 | 346.4 | 官方 cut_size=256 PAN。耗时脚本输入仍是 64×64 MS / 256 PAN，不是 OrigScale 128/512 |
| fusion_mamba | 420.ckpt | tile16_ov8 | test_hard | 49 | 0.735M | 60.09G | 2480.5 | 36.0 | 与 RDM-PAN 相同分块。耗时脚本输入仍是 64×64 MS / 256 PAN，不是 OrigScale 128/512 |
| rdm_pan | last.ckpt | tile16_ov8 | test | 49 | 0.625M | 103.96G | 2064.2 | 24.5 | RDM-PAN 分块 LR 16 / 重叠 8（PAN 64） |
| rdm_pan | last.ckpt | tile16_ov8 | test_hard | 49 | 0.625M | 103.96G | 2036.9 | 24.5 | 耗时脚本输入仍是 64×64 MS / 256 PAN，不是 OrigScale 128/512 |

引用 FLOPs 前先看 `tables/cost.md`。耗时脚本的 `test_hard` 行**不是** 512 PAN。

## 其它跑次怎么读

完整附录：`tables/all_runs.md` 和 `tables/all_runs.csv`。

会改数字的条件：

| 条件 | 含义 |
| --- | --- |
| FusionMamba RR `tile=1024 ov=0` | 256 整图，官方协议，对得上论文 RR 表。 |
| FusionMamba FR `tile=64 ov=0` + toolbox Ds | 官方 FR 切块。420 对得上论文 FR 表。 |
| FusionMamba FR 的 Ds ≈ 0.07 / QNR ≈ 0.91 | **Ds 协议错了**（torch 双三次）。不要拿去对论文。 |
| FusionMamba `tile=16 ov=8` | 强制跟 RDM 一样分块。RR 差不多；该目录的 FR Ds 用了错误协议。 |
| FusionMamba `tile=16 ov=0`（`compare_wv3_tile64_nooverlap`） | 无重叠分块。RR 会掉到约 39.02 / 38.96。 |
| RDM-PAN 所有完整指标行 | 一律 LR tile=16，重叠=8。CUDA Bi-WKV 不跑完整 256 序列。 |
| `compare_wv3_tiled` / 远程 `fusion_mamba/.../test` | 不完整（只有 PSNR 或 PSNR/SSIM）。不能当论文表。 |
| epoch 279 | 权重已删，只留下历史 FR 数字。 |

## 结论

- **RR：** RDM-PAN last 略好于官方 FusionMamba 420（PSNR 39.399 vs 39.343，SAM 2.782 vs 2.816，ERGAS 2.058 vs 2.079）。Q2n 基本打平（约 0.919）。
- **FR：** 仍是 FusionMamba 420 更好（QNR 0.9569 vs RDM 0.9385）。差距在 Dλ 和 Ds，不是 RR 失真。
- **速度：** 同一张 256 图，官方 FusionMamba 比分块 RDM-PAN 大约快 50 倍，因为 RDM 要跑 49 块。**同样** 16/8 分块时，RDM 还稍快一点（2.06 s vs 2.40 s），参数也更少（0.625M vs 0.735M）。
