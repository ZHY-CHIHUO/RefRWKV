# RefRWKV — Reference-Guided Remote Sensing Super-Resolution

基于 RWKV + Stable Diffusion 2.1 的参考引导遥感图像超分辨率重建框架，实现 10 倍超分（LR 48×48 → HR 480×480）。

## 核心链路

```
LR (48×48) ──→ RefSRWKV SR Prior ──→ SR image / SR latent ──┐
Ref (480×480) ──→ RWKV Adapter ──→ 多尺度残差 ─────────────┤
               ──→ DINOv2 + RWKV Pyramid ──→ 语义 token ────┤
                                                              ├──→ SD2 UNet (8ch) ──→ VAE ──→ HR (480×480)
              SR latent ──→ concat(noisy_latent) ────────────┘
```

## 架构组件

| 组件 | 作用 | 训练状态 |
|---|---|---|
| SD2 UNet + LoRA | 扩散去噪骨干；conv_in 扩展为 8 通道（4 noisy + 4 SR latent） | 🔥 可训练 |
| RWKV Adapter | 从 LR/Ref 提取多尺度特征注入 UNet | 🔥 可训练 |
| DINOv2 + RWKV Semantic Pyramid | 全局语义提取，注入 cross-attention | DINOv2 ❄️ / Pyramid 🔥 |
| RefSRWKV SR Prior | 直接超分，提供 SR latent 与结构先验 | ❄️ 冻结（四阶段） |
| D_sem (ConvNeXt/OpenCLIP) | 语义判别器 | 🔥 可训练 (Stage 4) |
| D_tex | 参考纹理一致性判别器（置信加权） | 🔥 可训练 (Stage 4) |

## 环境

- Python >= 3.10，PyTorch >= 2.1，CUDA >= 12.1
- 依赖：`diffusers transformers peft pytorch-lightning lpips pyiqa open_clip_torch vision_aided_loss lmdb pillow pyyaml tensorboard`
- RWKV WKV CUDA 算子通过 `torch.utils.cpp_extension.load` **首次运行时 JIT 编译**，无需手动 `setup.py install`（见 [WKV 后端](#wkv-后端)）。

## 数据准备

目录结构（默认 PNG 模式，主训练脚本使用 `RefPNGDataset`）：

```
<data_dir>/
├── train/{HR,LR,Ref}/*.png
├── val/{HR,LR,Ref}/*.png
└── test/{HR,LR,Ref}/*.png
```

- HR / Ref：480×480，LR：48×48（10× 下采样），文件名一一对应。
- 值域在数据集内部统一为 `[-1, 1]`。
- 裁剪先采 LR 整数坐标再乘 scale 映射 HR/Ref，保证严格对齐。

LMDB 加速（可选）：

```bash
python RefSR_data/convert_to_lmdb.py --data_dir RefSR_data/ALL_2
```

## 训练

### 配置结构（base + 覆盖）

所有训练/消融配置共用 `configs/base.yaml`（全部默认值：功能开关全关、loss 系数全 0），各配置通过 `base:` 字段引用并只写差异：

```
configs/
├── base.yaml                 # 公共默认（数据/训练器/模型默认/输出）
├── stage1_baseline.yaml      # Stage 1 覆盖
├── stage2_semantic.yaml      # Stage 2 覆盖
├── stage3_texture.yaml       # Stage 3 覆盖
├── stage4_gan.yaml           # Stage 4 覆盖
├── sr_prior.yaml             # SR Prior 独立训练
└── ablation/
    └── b_sd2_noref.yaml      # 消融 B：无参考（示例预置）
```

启动时打印「模块激活摘要」日志，开关与 loss 系数一目了然（消融对账用）。

### 四阶段课程

| Stage | 配置 | 新开启模块 | loss 系数 | lr / 精度 |
|---|---|---|---|---|
| 1 | `stage1_baseline.yaml` | —（扩散基线） | sr_noise=0.5 | 1e-5 / fp32 |
| 2 | `stage2_semantic.yaml` | 语义金字塔 + SR 条件 | sr_noise=1.0 | 1e-5 / bf16 |
| 3 | `stage3_texture.yaml` | SelfSim + 置信/时序门控 | diff_sr=0.3, lpips=0.5, sr_noise=0(warmdown) | 5e-6 / bf16 |
| 4 | `stage4_gan.yaml` | D_sem/D_tex + Swap + D_tex 加权 | +gan=0.02, gan_tex=0.05 | 1e-6 / fp32 |

### 关键开关速查

| 开关 | 默认 | 作用 |
|---|---|---|
| `model.use_reference` | true | false = 无参考消融（LR 上采样作中性自参考） |
| `model.use_semantic` | false | DINOv2 + RWKV 语义金字塔 |
| `model.use_sr_condition` | false | 语义金字塔的 SR latent 条件分支 |
| `model.rwkv_cfg.use_self_sim_transfer` | false | SR 自相似纹理迁移 |
| `model.use_confidence_gate` / `use_temporal_gate` | false | 置信 / 时序门控 |
| `model.use_discriminator` / `gan_enabled` | false | 双判别器 / GAN loss |
| `model.use_swap_test` / `dtex_conf_weight` | false | Swap Test / D_tex 置信加权 |
| `model.wkv_backend` | torch | 语义 WKV 后端（torch 默认，见下文） |

| 系数 | 默认 | 说明 |
|---|---|---|
| `lambda_diff_sr` | 0 | 像素/Latent 重建（Stage 3: 0.3） |
| `lambda_lpips` | 0 | 感知损失（Stage 3: 0.5） |
| `lambda_gan` / `lambda_gan_texture` | 0 | GAN 语义/纹理（Stage 4: 0.02 / 0.05） |
| `lambda_sr_noise` | 1.0 | SR 路径 ε 噪声权重 |

### 消融实验

论文消融矩阵与命令对照（相邻行差值 = 该模块贡献）：

| 实验 | 构成 | 命令 |
|---|---|---|
| A | 仅 RWKV SR | `python scripts/train_sr_prior.py --config configs/sr_prior.yaml` |
| B | +SD2 先验（无参考） | `python scripts/train_sd2_gan.py --config configs/ablation/b_sd2_noref.yaml` |
| C | +参考图 | `python scripts/train_sd2_gan.py --config configs/stage1_baseline.yaml` |
| D | +语义金字塔 | `python scripts/train_sd2_gan.py --config configs/stage2_semantic.yaml` |
| E | +GAN（完整） | `python scripts/train_sd2_gan.py --config configs/stage4_gan.yaml` |

子模块消融无需新配置，用 `--overrides` 即可（键支持点分路径）：

```bash
# 纹理消融：关 SelfSim
python scripts/train_sd2_gan.py --config configs/stage3_texture.yaml \
    --overrides model.rwkv_cfg.use_self_sim_transfer=false output.experiment_name=ab_no_selfsim

# 语义消融：关语义 + 关 LPIPS
python scripts/train_sd2_gan.py --config configs/stage4_gan.yaml \
    --overrides model.use_semantic=false model.lambda_lpips=0 output.experiment_name=ab_no_sem
```

恢复优先级：`--resume` > 配置 `resume_ckpt` > 实验目录 `last.ckpt` 自动检测 > 从头训练。跨阶段结构变化（如新增 semantic / discriminator）会自动回退为仅加载模型权重、optimizer 重新初始化。

## 评估

```bash
python evaluation/eval_pyiqa.py \
    --pred results/output/ --gt <data_dir>/test/HR/ \
    --fr_metrics psnr ssim lpips dists
```

指标方向：PSNR / SSIM ↑ 越高越好；LPIPS / DISTS ↓ 越低越好。

## WKV 后端

项目存在**两类** WKV 语义，需区分：

1. **空间 RWKV（SR Prior / Adapter）**：`models/RefSRWKV.py` 与 `models/RefDiffRWKV/RefDiffRWKV.py` 的 `VRWKV_SpatialMix` 直接调用 CUDA `bi_wkv` 算子，按 H→W 与 W→H 两种二维顺序各做一次（`recurrence=2`）。
2. **语义 RWKV（Semantic Pyramid）**：`globalsemanticmodule.py` 默认使用**纯 PyTorch 分块实现**（`wkv_backend="torch"`），通过「正向扫描 + 反向扫描取平均」实现双向。

**语义模块是否切回 CUDA？** 默认保持 `torch`，原因：

- 语义的 torch 实现是标准 RWKV-4 因果扫描（fp32、分块、数值护栏），语义明确、可复现、CPU 可运行；
- CUDA `bi_wkv` 算子是另一种**单遍双向**公式（同时累计过去 + 未来 + 当前 token），与 torch 的「两次单向取平均」**数值不等价**；
- 二者若混用会改变语义金字塔输出，进而改变 Stage 2+ 的训练轨迹。

如需使用 CUDA 后端，可在配置中显式 `wkv_backend: cuda`，但**必须先做数值对齐验证**（对比 torch 与 CUDA 的 forward/backward），确认通过后方可用于正式训练。

## 项目结构

```
RefRWKV/
├── configs/            # base + 四阶段 + sr_prior + ablation/
├── models/
│   ├── RefSRWKV.py     # SR Prior（含 WKV CUDA 延迟加载）
│   ├── cuda/           # bi_wkv.cpp / bi_wkv_kernel.cu
│   └── RefDiffRWKV/    # generator / adapter / semantic / discriminator / gan system / sampler
├── scripts/            # train_sd2_gan.py / train_sr_prior.py / test.py
├── evaluation/         # eval_pyiqa.py / eval_sewar.py
├── RefSR_data/         # RefDataset.py / convert_to_lmdb.py
└── checkpoints/        # 模型权重
```

## 训练监控指标

TensorBoard 指标一览（`logs/sd2_ref_gan/`）：

| 指标 | 含义 |
|---|---|
| `train/G_total` | G 总 loss |
| `train/G_diff_hr` | 扩散 ε-prediction（HR 路径） |
| `train/G_diff_sr` | 像素/Latent 重建 loss（lambda_diff_sr > 0 时） |
| `train/G_lpips` | 感知 loss（lambda_lpips > 0 时，降频计算） |
| `train/G_gan` | GAN loss（gan_enabled 时） |
| `train/D_sem` / `train/D_tex` | 判别器 loss（Stage 4） |
| `train/D_tex_conf` | D_tex 置信均值（健康区间 0.4~0.6） |
| `val/psnr` / `val/ssim` / `val/lpips` | 验证指标（LPIPS 越低越好） |

早停与最佳模型监控：`lambda_lpips > 0` 且 `fr_metrics` 含 lpips 时监控 `val/lpips`（min），否则监控 `val/psnr`（max）。

## 数据增强

- 空间增强（同步作用于 LR/HR/Ref）：随机水平/垂直翻转 + 90° 旋转（仅训练）。
- 参考图风格增强（仅 Ref）：随机灰度化（`ref_gray_prob`）、亮度/对比度/饱和度/色相扰动（`ref_aug_strengths` / `ref_aug_probs`），提升对参考图光照差异的鲁棒性。

## 故障排查

| 症状 | 可能原因 | 处理 |
|---|---|---|
| 启动报 `SR 权重无处加载` | `sr.ckpt_path` 不存在且无 `resume_ckpt` | 检查 `checkpoints/refrwkv_sr/best.ckpt` 或指定 `--resume` |
| 首次运行 JIT 编译失败 | CUDA/NVCC 版本与 GPU 不匹配 | Blackwell(sm_120) 需 CUDA >= 12.8 |
| `semantic_pyramid` 权重跳过 | WKV 公式与 checkpoint 不一致 | 属预期（公式变更），日志会提示 |
| 训练中断恢复 | — | 重交同一命令，`last.ckpt` 自动断点续训 |
| Stage 4 `D_tex_conf` 长期 < 0.2 | 数据对齐或参考增强过强 | `ref_aug_probs` 全 0 跑 100 步对比 |
| Stage 3 LPIPS 上升 | 纹理传播过强 | `self_sim_init_alpha` 降到 0.15 |
## 已知限制

- 仅支持 10× 超分（48→480），单卡 RTX 4090 24GB。
- CUDA WKV 算子需要 CUDA/NVCC，不支持纯 CPU 推理（空间 RWKV 路径）。
- `SelfSimTransfer` 全局 affinity 为 O(N²)，高分辨率下需注意显存（Stage 3/4 开启）。
- 数据加载器暂不支持断点续训的 epoch 内恢复（中断后重新遍历）。
- 评估暂未集成遥感专用指标（SAM / ERGAS）。

详见根目录 `优化.md` 获取优化明细与验证记录。
