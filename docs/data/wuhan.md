# Wuhan 数据集与 HRMS-SCD 迁移说明

## 网格与模型倍率

Wuhan 的 `L`（Landsat，30 m）和 `G`（Gaofen，8 m）文件虽然代表不同物理
分辨率，但发布文件已经配准并重采样为相同的 1000×1000 像素网格。因此
RefSRWKV 的输入/输出约定为：

```
L_t2 (lr, 4 bands) + G_t1 (ref, 4 bands) -> G_t2 (hr, 4 bands)
```

配置中的 `scale` 必须是 **1**。不能设置为 `30/8=3.75`，也不能为了满足
整数倍率而把 L 再缩小或把 G 放大；那会破坏已经完成的配准，并制造不对应的
监督目标。3.75 仅用于 ERGAS 的物理分辨率项（`100/3.75`）。

`WuhanSTFDataset` 仍可通过 `return_quadruple: true` 返回
`lr_t1/lr_t2/hr_t1/hr_t2`，便于以后接入真正的 STF 网络。

## 归一化与采样

TIFF 为四通道 `uint16`，按 `raw / 11848` 转为反射率 `[0,1]`，再映射为模型
使用的 `[-1,1]`。训练时四幅影像共享随机裁剪坐标和翻转/90°旋转；TIFF 读取
结果按绝对路径缓存在每个 Dataset 进程的字典中。数据量较小时推荐配置
`num_workers: 0`，避免每个 worker 各自复制缓存。

验证和测试保留完整 1000×1000 网格，但使用配置中的
`eval_tile_size: 128` / `eval_tile_overlap: 16` 做重叠拼接推理。这样不会把
L 或 G 重新缩放，且能避免 RefSRWKV 在全图局部匹配 `unfold` 时的显存峰值。
显存充足时可以增加 tile size；不要将 overlap 设为不小于 tile size。

## HRMS-SCD 预训练迁移

方案可行，但不是 checkpoint 原样加载：

1. HRMS-SCD 是 3 通道、x4 的 RefSR；Wuhan 是 4 通道、同网格 x1 的时空融合。
2. 配置 Wuhan 模型为 `inp_channels=ref_channels=out_channels=4, scale=1`。
3. 使用 `channel_adaptation: rgb_mean` 时，兼容的输入/参考 RGB 边界卷积会
   复制前三个通道，第四通道用前三通道均值初始化；PixelUnshuffle、重建倍率
   相关层和 STF 融合头会跳过并保持 Wuhan 的新初始化。
4. 建议先冻结可迁移骨干若干 epoch，再整体解冻；同时保留 Wuhan 从头训练
   作为公平基线。若 checkpoint 来自 STFMamba 而不是本仓库同构的 RefSRWKV，
   不能按参数名迁移，只能另行设计特征级蒸馏/转换。

## STFMamba 对照注意事项

检查 `zhaomin0101/STFMamba` 的公开代码可见，其 `model.py` 将
`MambaSR`、`STFMamba` 与 `Cross_MultiAttention` 的输入通道硬编码为 6，且其
`PatchSet` 读取的是预先打包的 `.npy`。因此该仓库不能直接把本数据集的四通道
TIFF 喂进去；后续接入时应将这些 6 通道参数统一改为 4，并使用本仓库返回的
`lr_t1/lr_t2/hr_t1/hr_t2` 四元组。不要为了兼容原代码而丢弃第四个波段。
公开训练循环的 STF 参数可对应为 `ref_lr=lr_t1`、`data=lr_t2`、
`ref_target=hr_t1`、`target=hr_t2`。
