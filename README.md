# RefRWKV — Reference-Guided Remote Sensing Super-Resolution

基于 RWKV + Stable Diffusion 2 的参考引导遥感图像超分辨率重建框架。

## 核心思路

将参考图像（Reference）中的纹理和语义信息注入扩散模型，实现 10 倍遥感图像超分辨率重建：

```
LR (48×48) ──→ SR Prior (RefSRWKV) ──→ SR Latent ──┐
                                                     ├──→ SD2 UNet ──→ HR (480×480)
Ref (480×480) ──→ Adapter (RWKV) ──→ 多尺度残差 ────┤
              ──→ DINOv2 + RWKV Pyramid ──→ 语义 Token ──→ Cross-Attention
```

## 架构组件

| 组件 | 作用 | 参数量 | 训练状态 |
|---|---|---|---|
| **SD2 UNet + LoRA** | 扩散去噪骨干网络 | ~200M (LoRA ~20M) | 🔥 可训练 |
| **Adapter (RWKV)** | 从参考图提取多尺度特征，注入 UNet 各层 | ~10M | 🔥 可训练 |
| **DINOv2 + RWKV Pyramid** | 全局语义提取，注入 Cross-Attention | ~100M (Pyramid ~16M) | DINOv2 ❄️ / Pyramid 🔥 |
| **SR Prior (RefSRWKV)** | 直接超分，提供初始 latent 和 DPS 引导 | ~25M | ❄️ 冻结 |
| **D_sem (ConvNeXt)** | 语义判别器：图像是否像真实遥感图 | ~88M | 🔥 可训练 (Phase2) |
| **D_tex** | 纹理一致性判别器：纹理是否与参考图一致 | ~7M | 🔥 可训练 (Phase2) |

## 训练流程

### Phase 1：扩散训练（无 GAN）

```yaml
gan_enabled: false
```

- **Loss**：ε-prediction MSE（预测噪声）
- **目标**：学习从加噪 latent 恢复清晰图像
- **监控**：`val_psnr`（max）

### Phase 2：GAN 精调

```yaml
gan_enabled: true
```

- **Loss**：
  - 扩散 ε-prediction MSE（始终）
  - 像素 MSE：`MSE(pred_sr_pixel, hr)`（`lambda_diff_sr`）
  - 感知 LPIPS：`LPIPS(pred_sr_pixel, hr)`（`lambda_lpips`）
  - GAN 语义：`-D_sem(fake)`（`lambda_gan`）
  - GAN 纹理：`-D_tex(fake, ref)`（`lambda_gan_texture`）
- **D 预热**：前 `d_warmup_steps` 步只训练判别器（默认 3000）
- **监控**：`val/lpips`（min）

> **注意**：LPIPS 和像素损失由对应的 lambda 系数控制（系数 > 0 即启用），无需额外开关。只有判别器需要独立的 `gan_enabled` 结构性开关。

### 推理（Better Start + DPS Guidance）

```
SR Prior 输出 → add_noise(t_start) → 从 t_start 去噪 → DPS 引导向 SR 结果靠近 → 输出
```

- 不从纯噪声开始，而是从 SR Prior 的结果加部分噪声开始
- DPS (Diffusion Posterior Sampling) 引导生成向 SR 结果靠近

## 安装

```bash
# 环境要求
Python >= 3.10
PyTorch >= 2.1
CUDA >= 12.1

# 依赖
pip install diffusers transformers peft pytorch-lightning lpips pyiqa
pip install open_clip_torch vision_aided_loss
pip install lmdb pillow pyyaml tensorboard

# CUDA 扩展（RWKV WKV 算子）
cd models/cuda
python setup.py install
```

## 数据准备

### 目录结构

```
data_dir/
├── train/
│   ├── HR/    (480×480 PNG)
│   ├── LR/    (48×48 PNG)
│   └── Ref/   (480×480 PNG)
├── val/
│   ├── HR/
│   ├── LR/
│   └── Ref/
└── test/
    ├── HR/
    ├── LR/
    └── Ref/
```

- **HR**：高分辨率真值（480×480）
- **LR**：低分辨率输入（48×48，10 倍下采样）
- **Ref**：参考图像（480×480，与 HR 同场景不同时间/角度）
- 值域：`[-1, 1]`（数据集内部自动转换）

### LMDB 加速（可选）

```bash
python RefSR_data/convert_to_lmdb.py --data_dir data_dir --mode train
```

## 训练

### Phase 1：扩散训练

```bash
python scripts/train_sd2_gan.py --config configs/sd2_ref_gan_config.yaml
```

配置要点：

```yaml
model:
  gan_enabled: false          # 不启用 GAN
  lambda_diff_sr: 0.0         # 关闭辅助 loss
  lambda_lpips: 0.0
  lambda_gan_semantic: 0.0
  lambda_gan_texture: 0.0
```

### Phase 2：GAN 精调

```bash
python scripts/train_sd2_gan.py \
    --config configs/sd2_ref_gan_config.yaml \
    --resume checkpoints/sd2_ref_gan_Phase1/best_psnr.ckpt
```

配置要点：

```yaml
model:
  gan_enabled: true           # 启用 GAN
  d_warmup_steps: 3000        # D 预热步数
  lambda_diff_sr: 0.3         # 像素 loss
  lambda_lpips: 0.5           # 感知 loss
  lambda_gan_semantic: 0.02   # GAN 语义 loss
  lambda_gan_texture: 0.05    # GAN 纹理 loss

train:
  resume_ckpt: null           # 或指定 checkpoint 路径
  best_save:
    metrics: ["psnr", "ssim", "lpips"]
    min_improved: 2           # 至少 2 项同时改进才保存
```

### 恢复策略

```
优先级：--resume 命令行 > 配置 resume_ckpt > 从头训练
```

## 评估

```bash
python evaluation/eval_pyiqa.py \
    --pred results/output/ \
    --gt data_dir/test/HR/ \
    --fr_metrics psnr ssim lpips dists
```

### 指标说明

| 指标 | 含义 | 方向 |
|---|---|---|
| PSNR (Y-channel) | 峰值信噪比（亮度通道） | ↑ 越高越好 |
| SSIM (Y-channel) | 结构相似度（亮度通道） | ↑ 越高越好 |
| LPIPS | 感知相似度 | ↓ 越低越好 |
| DISTS | 深度图像结构纹理相似度 | ↓ 越低越好 |

## 配置说明

### 关键参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `gan_enabled` | `false` | GAN 训练开关（结构性，独立于损失系数） |
| `d_warmup_steps` | `3000` | D 预热步数（仅 `gan_enabled=True` 时生效） |
| `learning_rate` | `1e-4` | Generator 学习率 |
| `lr_D_sem` | `5e-6` | D_sem 学习率 |
| `lr_D_tex` | `1e-6` | D_tex 学习率 |
| `accumulate_grad_batches` | `8` | 梯度累积步数 |
| `t_min` / `t_max` | `300` / `700` | 训练时间步范围 |
| `t_start` | `300` | 推理起始时间步 |
| `sample_steps` | `50` | 推理去噪步数 |
| `grad_clip_val` | `1.0` | 梯度裁剪阈值 |
| `cfg_drop_prob` | `0.1` | CFG dropout 概率 |
| `lora_rank` | `64` | LoRA 秩 |

### 训练监控指标

| 指标 | 含义 |
|---|---|
| `train/G_total` | G 总 loss |
| `train/G_diff_hr` | 扩散 ε-prediction loss |
| `train/G_diff_sr` | 像素 MSE loss |
| `train/G_lpips` | LPIPS 感知 loss |
| `train/G_gan` | GAN 总 loss |
| `train/D_sem` | 语义判别器 loss |
| `train/D_tex` | 纹理判别器 loss |
| `val/psnr` | 验证 PSNR |
| `val/lpips` | 验证 LPIPS |

## 项目结构

```
RefRWKV/
├── configs/
│   └── sd2_ref_gan_config.yaml       # 训练配置
├── models/
│   ├── RefSRWKV.py                    # SR Prior（RWKV 超分模型）
│   ├── cuda/                          # RWKV WKV CUDA 算子
│   └── RefDiffRWKV/
│       ├── sd2_ref_generator.py       # 扩散生成器
│       ├── sd2_ref_discriminator.py   # 双判别器
│       ├── sd2_ref_gan_system.py      # GAN 训练系统
│       ├── sd2_ref_adapter.py         # 参考图 Adapter
│       └── globalsemanticmodule.py    # DINOv2 + RWKV 语义模块
├── scripts/
│   └── train_sd2_gan.py              # 训练脚本
├── evaluation/
│   ├── eval_pyiqa.py                 # IQA 评估
│   └── eval_sewar.py                 # 遥感指标评估
├── RefSR_data/
│   ├── RefDataset.py                 # 数据集
│   └── convert_to_lmdb.py            # LMDB 转换
└── checkpoints/                       # 模型权重
```

## 关键设计决策

| 决策 | 原因 |
|---|---|
| UNet conv_in 4→8 通道 | 后 4 通道输入 SR latent 条件，前 4 通道保留预训练权重 |
| 前 4 通道梯度 hook 屏蔽 | 防止破坏 SD2 预训练权重 |
| DINOv2 冻结，Pyramid 可训练 | DINOv2 特征已足够好，Pyramid 需要适配遥感任务 |
| Better Start（从 SR latent 加噪） | 比从纯噪声开始更快收敛，质量更好 |
| DPS Guidance | 引导生成向 SR Prior 结果靠近，保持结构一致性 |
| 双判别器（语义 + 纹理） | 语义 D 保证整体质量，纹理 D 保证参考图纹理迁移 |
| D 预热 `d_warmup_steps` 步 | 让随机初始化的 D 先学会基本区分，再与 G 对抗 |
| GAN 与 LPIPS 解耦 | D 爆炸不影响感知损失的反向传播，提升训练稳定性 |
| 验证固定 seed=42 | 保证跨 epoch 验证结果可比较 |

## 已知限制

- 仅支持 10 倍超分（48×48 → 480×480）
- 单卡训练（RTX 4090 24GB）
- 评估指标以 Y-channel PSNR/SSIM 为主，遥感专用指标（SAM/ERGAS）待集成
- CUDA WKV 算子需要编译，不支持纯 CPU 推理
