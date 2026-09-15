# 磁盘上全部 PanCollection WV3 `metrics.json`

包含 RDM-PAN 中途权重、自训 FusionMamba `last.ckpt`、官方 `420.ckpt`，以及不完整的 PSNR 行。STF / 武汉数据不在这里。

重复项：`rdm_pan_last_ov8`（若还在）等于 `rdm_pan_ckpts/epoch0439`；`compare_wv3_fm_official/fusion_mamba_420` 的 RR 等于 `compare_wv3_fullimage/fusion_mamba_420` 的 RR；`fusion_mamba_420_qnr_toolbox` 的 FR 等于官方 420 的 toolbox Ds；`compare_wv3_pan/fusion_mamba` 的 last FR 等于 `compare_wv3_fm_official/fusion_mamba_last` 的 FR。

| 跑次目录 | 模型 | 划分 | 权重 | Tile | 重叠 | 推理方式 | 是否完整 | Ds 协议 | PSNR | Q2n | SAM° | ERGAS | SSIM | Dλ | Ds | QNR |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| compare_wv3_fm_official/fusion_mamba_420/test | fusion_mamba | test | 420.ckpt | 1024 | 0 | 整图推理（tile=1024 对 64×64 LR / 256 PAN 等于没切） | 完整 RR | 无 | 39.3434±2.8183 | 0.9192±0.0814 | 2.8163±0.5229 | 2.0795±0.4580 |  |  |  |  |
| compare_wv3_fm_official/fusion_mamba_420/test_hard | fusion_mamba | test_hard | 420.ckpt | 64 | 0 | 官方 FR 切块：LR tile=64，重叠=0（PAN 256） | 完整 FR | toolbox MATLAB imresize（与论文一致） |  |  |  |  |  | 0.0166±0.0057 | 0.0270±0.0059 | 0.9569±0.0110 |
| compare_wv3_fm_official/fusion_mamba_last/test | fusion_mamba | test | last.ckpt | 1024 | 0 | 整图推理（tile=1024 对 64×64 LR / 256 PAN 等于没切） | 完整 RR | 无 | 39.2891±2.8166 | 0.9182±0.0817 | 2.8277±0.5221 | 2.0910±0.4554 |  |  |  |  |
| compare_wv3_fm_official/fusion_mamba_last/test_hard | fusion_mamba | test_hard | last.ckpt | 64 | 0 | 官方 FR 切块：LR tile=64，重叠=0（PAN 256） | 完整 FR | toolbox MATLAB imresize（与论文一致） |  |  |  |  |  | 0.0189±0.0048 | 0.0339±0.0032 | 0.9479±0.0073 |
| compare_wv3_fullimage/fusion_mamba/test | fusion_mamba | test | last.ckpt | 1024 | 0 | 整图推理（tile=1024 对 64×64 LR / 256 PAN 等于没切） | 完整 RR | 无 | 39.2891±2.8166 | 0.9182±0.0817 | 2.8277±0.5221 | 2.0910±0.4554 |  |  |  |  |
| compare_wv3_fullimage/fusion_mamba/test_hard | fusion_mamba | test_hard | last.ckpt | 1024 | 0 | OrigScale 尝试整图（tile=1024，重叠=0） | 完整 FR | torch 双三次 Ds（协议不对，Ds 偏大） |  |  |  |  |  | 0.0186±0.0049 | 0.0664±0.0326 | 0.9164±0.0350 |
| compare_wv3_fullimage/fusion_mamba_420/test | fusion_mamba | test | 420.ckpt | 1024 | 0 | 整图推理（tile=1024 对 64×64 LR / 256 PAN 等于没切） | 完整 RR | 无 | 39.3434±2.8183 | 0.9192±0.0814 | 2.8163±0.5229 | 2.0795±0.4580 |  |  |  |  |
| compare_wv3_fullimage/fusion_mamba_420/test_hard | fusion_mamba | test_hard | 420.ckpt | 64 | 0 | 官方 FR 切块：LR tile=64，重叠=0（PAN 256） | 完整 FR | torch 双三次 Ds（协议不对，Ds 偏大） |  |  |  |  |  | 0.0166±0.0057 | 0.0740±0.0353 | 0.9108±0.0386 |
| compare_wv3_fullimage/fusion_mamba_420_qnr_toolbox/test_hard | fusion_mamba | test_hard | 420.ckpt | 64 | 0 | 官方 FR 切块：LR tile=64，重叠=0（PAN 256） | 完整 FR | toolbox MATLAB imresize（与论文一致） |  |  |  |  |  | 0.0166±0.0057 | 0.0270±0.0059 | 0.9569±0.0110 |
| compare_wv3_official/fusion_mamba/test | fusion_mamba | test | last.ckpt | 16 | 8 | RDM 分块：LR tile=16，重叠=8（PAN 64，256 图共 49 块） | 完整 RR | 无 | 39.3399±2.8507 | 0.9188±0.0816 | 2.7936±0.5198 | 2.0727±0.4413 |  |  |  |  |
| compare_wv3_official/fusion_mamba/test_hard | fusion_mamba | test_hard | last.ckpt | 16 | 8 | RDM 分块：LR tile=16，重叠=8（PAN 64，256 图共 49 块） | 完整 FR | torch 双三次 Ds（协议不对，Ds 偏大） |  |  |  |  |  | 0.0194±0.0047 | 0.0654±0.0310 | 0.9166±0.0332 |
| compare_wv3_pan/fusion_mamba/test | fusion_mamba | test | last.ckpt | 1024 | 0 | 整图推理（tile=1024 对 64×64 LR / 256 PAN 等于没切） | 完整 RR | 无 | 39.2891±2.8166 | 0.9182±0.0817 | 2.8277±0.5221 | 2.0910±0.4554 |  |  |  |  |
| compare_wv3_pan/fusion_mamba/test_hard | fusion_mamba | test_hard | last.ckpt | 64 | 0 | 官方 FR 切块：LR tile=64，重叠=0（PAN 256） | 完整 FR | toolbox MATLAB imresize（与论文一致） |  |  |  |  |  | 0.0189±0.0048 | 0.0339±0.0032 | 0.9479±0.0073 |
| compare_wv3_tile64_nooverlap/fusion_mamba_420/test | fusion_mamba | test | 420.ckpt | 16 | 0 | 无重叠分块：LR tile=16，重叠=0（PAN 64） | 完整 RR | 无 | 39.0171±2.8804 | 0.9162±0.0824 | 2.8575±0.5379 | 2.1591±0.4643 |  |  |  |  |
| compare_wv3_tile64_nooverlap/fusion_mamba_420/test_hard | fusion_mamba | test_hard | 420.ckpt | 16 | 0 | 无重叠分块：LR tile=16，重叠=0（PAN 64） | 完整 FR | toolbox MATLAB imresize（与论文一致） |  |  |  |  |  | 0.0170±0.0053 | 0.0264±0.0048 | 0.9571±0.0096 |
| compare_wv3_tile64_nooverlap/fusion_mamba_last/test | fusion_mamba | test | last.ckpt | 16 | 0 | 无重叠分块：LR tile=16，重叠=0（PAN 64） | 完整 RR | 无 | 38.9565±2.8940 | 0.9152±0.0825 | 2.8669±0.5353 | 2.1709±0.4589 |  |  |  |  |
| compare_wv3_tile64_nooverlap/fusion_mamba_last/test_hard | fusion_mamba | test_hard | last.ckpt | 16 | 0 | 无重叠分块：LR tile=16，重叠=0（PAN 64） | 完整 FR | toolbox MATLAB imresize（与论文一致） |  |  |  |  |  | 0.0197±0.0045 | 0.0345±0.0031 | 0.9465±0.0069 |
| compare_wv3_tiled/fusion_mamba/test | fusion_mamba | test | last.ckpt | 16 | 8 | RDM 分块：LR tile=16，重叠=8（PAN 64，256 图共 49 块） | 仅 PSNR/SSIM（不能当论文表） | 无 | 39.33989944 |  |  |  | 0.9757742047 |  |  |  |
| fusion_mamba/pancollection_wv3/x4/fusion_mamba_pancollection_wv3_official/test | fusion_mamba | test | last.ckpt |  |  | 未指定分块（默认整图） | 仅 PSNR/SSIM（不能当论文表） | 无 | 39.28898659 |  |  |  | 0.9754913151 |  |  |  |
| rdm_pan_ckpts/epoch0409/test | rdm_pan | test | epoch=0409-step=124230 | 16 | 8 | RDM 分块：LR tile=16，重叠=8（PAN 64，256 图共 49 块） | 完整 RR | 无 | 39.3988±2.8499 | 0.9194±0.0821 | 2.7830±0.5267 | 2.0581±0.4359 | 0.9761±0.0074 |  |  |  |
| rdm_pan_ckpts/epoch0409/test_hard | rdm_pan | test_hard | epoch=0409-step=124230 | 16 | 8 | RDM 分块：LR tile=16，重叠=8（PAN 64，256 图共 49 块） | 完整 FR | toolbox MATLAB imresize（与论文一致） |  |  |  |  |  | 0.0265±0.0075 | 0.0395±0.0038 | 0.9350±0.0104 |
| rdm_pan_ckpts/epoch0419/test | rdm_pan | test | epoch=0419-step=127260 | 16 | 8 | RDM 分块：LR tile=16，重叠=8（PAN 64，256 图共 49 块） | 完整 RR | 无 | 39.3960±2.8502 | 0.9195±0.0817 | 2.7817±0.5264 | 2.0591±0.4357 | 0.9761±0.0075 |  |  |  |
| rdm_pan_ckpts/epoch0419/test_hard | rdm_pan | test_hard | epoch=0419-step=127260 | 16 | 8 | RDM 分块：LR tile=16，重叠=8（PAN 64，256 图共 49 块） | 完整 FR | toolbox MATLAB imresize（与论文一致） |  |  |  |  |  | 0.0245±0.0066 | 0.0381±0.0034 | 0.9384±0.0091 |
| rdm_pan_ckpts/epoch0439/test | rdm_pan | test | epoch=0439-step=133320 | 16 | 8 | RDM 分块：LR tile=16，重叠=8（PAN 64，256 图共 49 块） | 完整 RR | 无 | 39.3994±2.8494 | 0.9193±0.0820 | 2.7821±0.5257 | 2.0580±0.4346 | 0.9760±0.0075 |  |  |  |
| rdm_pan_ckpts/epoch0439/test_hard | rdm_pan | test_hard | epoch=0439-step=133320 | 16 | 8 | RDM 分块：LR tile=16，重叠=8（PAN 64，256 图共 49 块） | 完整 FR | toolbox MATLAB imresize（与论文一致） |  |  |  |  |  | 0.0249±0.0064 | 0.0375±0.0029 | 0.9385±0.0085 |
