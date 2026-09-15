# 全分辨率（FR）— PanCollection WV3 `test_hard`

- 划分：20 张 OrigScale，LR MS `8×128×128`，PAN `1×512×512`，没有 HR 真值。
- 指标：Dλ / Ds / QNR。
- **表头 Ds 必须用 toolbox 的 MATLAB `imresize` 协议。** torch 双三次 Ds（大约 0.07）协议不对，只放在全部跑次附录里。
- FusionMamba 官方 FR 切块是 LR tile=64 / 重叠=0（PAN 256）。RDM-PAN 是 LR tile=16 / 重叠=8。

| 模型 | 权重 | 推理方式 | Dλ↓ | Ds↓ | QNR↑ | Ds 协议 |
| --- | --- | --- | --- | --- | --- | --- |
| 论文 FusionMamba | 论文表格 | 官方（论文） | 0.0186±0.0078 | 0.0269±0.0058 | 0.9550±0.0110 | 论文 |
| FusionMamba 官方 420 | 420.ckpt | LR tile=64，重叠=0 | 0.0166±0.0057 | 0.0270±0.0059 | 0.9569±0.0110 | toolbox MATLAB imresize（与论文一致） |
| FusionMamba 自训 last | last.ckpt | LR tile=64，重叠=0 | 0.0189±0.0048 | 0.0339±0.0032 | 0.9479±0.0073 | toolbox MATLAB imresize（与论文一致） |
| RDM-PAN last（= epoch 439） | last.ckpt | LR tile=16，重叠=8 | 0.0249±0.0064 | 0.0375±0.0029 | 0.9385±0.0085 | toolbox MATLAB imresize（与论文一致） |
| RDM-PAN epoch 419 | epoch=0419-step=127260 | LR tile=16，重叠=8 | 0.0245±0.0066 | 0.0381±0.0034 | 0.9384±0.0091 | toolbox MATLAB imresize（与论文一致） |
| RDM-PAN epoch 409 | epoch=0409-step=124230 | LR tile=16，重叠=8 | 0.0265±0.0075 | 0.0395±0.0038 | 0.9350±0.0104 | toolbox MATLAB imresize（与论文一致） |
| RDM-PAN epoch 279（已删） | epoch=0279（不在盘上） | LR tile=16，重叠=8 | 0.0219 | 0.0378 | 0.9412 | 历史 FR 记录，权重已删 |

epoch 279 权重已经不在磁盘上，FR 数字来自之前的评测日志。
