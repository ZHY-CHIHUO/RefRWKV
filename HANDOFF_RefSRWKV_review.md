# RefSRWKV 审查与交接记录

## 当前工作区

- Windows 工作副本：`C:\Users\ZHY\Desktop\RefRWKV`
- 分支：`dev`
- 远端：`origin` (`https://github.com/ZHY-CHIHUO/RefRWKV.git`)
- 本次范围：`models/RefSRWKV.py`、`scripts/train_sr_prior.py`
- 基线提交：`43c181a`
- 源码和本文件均按 UTF-8 保存；PowerShell 查看源码时使用
  `Get-Content -Encoding UTF8`，否则中文可能显示为乱码。

## 已完成修改

### `models/RefSRWKV.py`

1. 修复 shifted-window 的 padding 计算。原实现没有把 shift 计入总尺寸，部分分辨率会在 `einops` 切窗时直接崩溃；现在先在上/左侧补 shift，再在下/右侧补到 8 的倍数，最后裁回原尺寸。
2. 修复 `GatedFusion` 漏乘融合特征的问题。参考特征现在实际通过 `gate * confidence * fused` 注入 LR 分支。
3. 修复 WKV 衰减参数符号。CUDA kernel 使用 `exp(-w * distance)`，参数改为 inverse-softplus 初始化并在调用前 softplus，保证距离衰减为正。
4. 恢复最终的高分辨率输出路径：`HR/4 -> PixelShuffle(2) -> PixelShuffle(2) -> output_conv`。`output_conv` 零初始化，训练初始输出接近 bicubic skip；最终使用 `clamp([-1, 1])`，避免原 `tanh` 把 bicubic 基线压缩。
5. bicubic skip 增加显式 `skip_proj`。默认 RGB (`3 -> 3`) 是 `Identity`，不同输入/输出通道时使用 1x1 投影；参考图仍要求 `ref_channels == inp_channels`，因为颜色统计匹配以 LR 为目标。
6. 加强输入、尺寸、通道和构造参数检查；固定参考折叠因子为 4，并要求训练 HR patch 可被 32 整除。
7. `_match_color` 使用 float32、`unbiased=False` 和标准差下限，避免半精度/常数 patch 产生 NaN；参考金字塔改为逐级计算，减少重复卷积。
8. 修复 `OmniShift` 重参数化时的 dtype/device 不一致。
9. 训练封装修复：pyiqa 使用 `as_loss=True` 且显式转换到 `[0,1]`；手写 SSIM 使用正确的二维 Gaussian depthwise kernel；FFT loss 不再重复计算；test 只 forward 一次。
10. EMA 只在真实 optimizer step 后更新，兼容梯度累积；验证/测试套用和恢复 EMA 的 hook 幂等；warmup 为 0 或训练步数不足时不创建非法 `LinearLR`；梯度裁剪保留 `0` 表示禁用。

### `scripts/train_sr_prior.py`

1. 配置、数据、日志和 checkpoint 的相对路径统一锚定仓库根目录，支持从 PowerShell 任意当前目录启动；配置读取兼容 UTF-8 BOM。
2. `data.scale` 作为唯一倍率来源，并校验 `model.scale` 一致、HR patch 可被 32 整除、通道/训练参数/数据增强参数合法。
3. Windows `DataLoader` 兼容性修复：`num_workers=0` 时不传 `prefetch_factor`/`persistent_workers`；空数据集和 `drop_last` 导致零 step 会提前报错或自动处理。
4. 支持配置的 batch key、采样 seed、accelerator/devices/precision 等选项。
5. `--load_weights` 与 `--resume` 互斥；checkpoint loader 支持多层 wrapper 前缀和多种嵌套 state dict；热启动只加载 shape 匹配参数，不会自动改名或移动不兼容的 `last.ckpt`。
6. 删除重复的强制保存 callback，使用 `ModelCheckpoint(save_last=True)`；支持可选 `max_steps`、`strategy` 和 sanity validation 配置。

## 当前尺寸路径

模型不是把 LR 直接按倍率逐层放大，而是固定在 `internal_size = hr_size / 4` 上运行：

```
LR (P/scale) --bicubic--> internal LR (P/4)
Ref (P) --PixelUnshuffle(4)--> level 1 (P/4)
level 1 --3 x PixelUnshuffle(2)--> P/8 -> P/16 -> P/32
decoder --3 x PixelShuffle(2)--> d1 (P/4)
d1 --2 x PixelShuffle(2)--> HR residual (P)
output = clamp(bicubic(LR, Ref size) + residual, -1, 1)
```

因此训练 patch 的 HR 边长必须是 32 的倍数。现有配置的尺寸为：

| 配置 | HR patch | LR patch | scale | internal |
| --- | ---: | ---: | ---: | ---: |
| `sr_prior_4.yaml` | 512 | 128 | 4 | 128 |
| `sr_prior_10.yaml` | 480 | 48 | 10 | 120 |

`scale` 影响数据集 crop 和记录信息；模型内部始终使用固定的 HR/4 网格。

## 数据集约定

当前训练脚本使用 `RefPNGDataset`，目录必须为：

```
<root>/<mode>/LR/*.png
<root>/<mode>/HR/*.png
<root>/<mode>/Ref/*.png
```

三个目录按同名 PNG 配对。图像读取为 RGB，并归一化到 `[-1, 1]`。随机 crop 先在 LR 上抽整数坐标，再乘 `scale` 映射到 HR/Ref，保证三者空间对齐；训练阶段可对参考图做颜色/灰度增强。

## 已执行验证

Windows 环境已通过：

```text
python -m py_compile models/RefSRWKV.py scripts/train_sr_prior.py
UTF-8 compile（两个文件）
git diff --check
```

此机器没有安装 `torch`、`einops`、`pytorch_lightning` 或 CUDA，因此没有声称完成真实 forward/backward、Lightning 或 CUDA kernel 验证。

## WSL 端下一步

1. 拉取 `dev` 最新提交，确认安装 CUDA 版 PyTorch、`einops`、Lightning、`torchvision`、`Pillow`、`lmdb` 和可选 `pyiqa`。
2. 先用缩小 block/尺寸配置做 GPU forward；至少覆盖 batch=1、batch=2、`hr_size=512`、`hr_size=480`、参考图 dropout、SSIM、`warmup_steps=0`。
3. 编译并测试 `models/cuda/bi_wkv`：检查 `T=64`、非 32 倍的 T、forward/backward 输出 finite，并做小规模梯度数值检查。
4. 用真实数据分别运行 `sr_prior_4.yaml` 和 `sr_prior_10.yaml` 的短 smoke run，确认 DataLoader crop、输出 shape、EMA 和 checkpoint 保存/恢复。
5. 对旧 SR checkpoint 使用 `--load_weights` 热启动；由于新增 `up_final/output_conv`，旧 checkpoint 不应强制使用 `--resume`。
6. 重新检查 `scripts/train_sd2_gan.py`、评测脚本构造 RefSRWKV 时的 `hr_size` 与训练配置是否一致，确认新输出头权重被正确读取。尤其是 `sr_prior_4` 在 512 patch 上训练时，下游若仍使用默认 `hr_size=480`，内部网格会不同，需按实验意图明确设置。
7. 在确认数值和指标后，再决定是否把 downstream loader、评测脚本或文档中的旧输出头/`tanh` 描述同步更新。

## 交接注意事项

- 不要把 Windows 端“编译通过”当作 CUDA smoke test 结论。
- 若出现中文乱码，优先确认终端读取编码，不要据此修改源码编码。
- 目前没有修改数据集实现、`train_sd2_gan.py` 或评测脚本；这些属于后续 WSL 验证和联调范围。
