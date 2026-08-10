以下是修改后的完整 README，在末尾追加了**改进路线图**部分：

```markdown
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

### Stage 1：基础扩散训练（无 GAN，无语义）

```yaml
gan_enabled: false
use_semantic: false
use_sr_condition: false
```

- **Loss**：ε-prediction MSE（预测噪声）
- **目标**：学习基础去噪 + Adapter 特征注入
- **监控**：`val_psnr`（max）

### Stage 2：语义路径重训（新 WKV 公式）+ SR 条件分支

```yaml
use_semantic: true
use_sr_condition: true
gan_enabled: false
```

- **新增模块**：DINOv2 + RWKV Pyramid（随机初始化）、SR 条件分支
- **Loss**：ε-prediction MSE
- **目标**：学习语义引导 + SR 结构约束
- **恢复**：从 Stage 1 跨阶段恢复（仅模型权重，optimizer 重新初始化）

### Stage 3：纹理精调

```yaml
use_semantic: true
use_sr_condition: true
use_confidence_gate: true
gan_enabled: false
```

- **新增模块**：置信度门控（Confidence Gate）
- **Loss**：ε-prediction MSE + 像素 MSE（`lambda_diff_sr`）+ LPIPS（`lambda_lpips`）
- **目标**：学习纹理迁移 + 感知质量提升

### Stage 4：GAN 精调

```yaml
gan_enabled: true
use_discriminator: true
```

- **Loss**：
  - 扩散 ε-prediction MSE（始终）
  - 像素 MSE：`MSE(pred_sr_pixel, hr)`（`lambda_diff_sr`）
  - 感知 LPIPS：`LPIPS(pred_sr_pixel, hr)`（`lambda_lpips`）
  - GAN 语义：`-D_sem(fake)`（`lambda_gan`）
  - GAN 纹理：`-D_tex(fake, ref)`（`lambda_gan_texture`）
- **D 预热**：前 `d_warmup_steps` 步只训练判别器
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

### Stage 1：基础扩散训练

```bash
python scripts/train_sd2_gan.py --config configs/stage1_baseline.yaml
```

### Stage 2：语义路径重训

```bash
python scripts/train_sd2_gan.py --config configs/stage2_semantic.yaml
```

配置要点：

```yaml
train:
  resume_ckpt: null           # null 时自动检测当前实验目录的 last.ckpt
```

### Stage 3：纹理精调

```bash
python scripts/train_sd2_gan.py --config configs/stage3_texture.yaml
```

### Stage 4：GAN 精调

```bash
python scripts/train_sd2_gan.py --config configs/stage4_gan.yaml
```

配置要点：

```yaml
model:
  gan_enabled: true
  d_warmup_steps: 3000
  lambda_diff_sr: 0.3
  lambda_lpips: 0.5
  lambda_gan_semantic: 0.02
  lambda_gan_texture: 0.05

train:
  resume_ckpt: null
  best_save:
    metrics: ["psnr", "ssim", "lpips"]
    min_improved: 2
```

### 恢复策略

```
优先级：--resume 命令行 > 配置 resume_ckpt > 自动检测 last.ckpt > 从头训练
```

跨阶段恢复时，预检测 optimizer 参数数量：
- 匹配 → 完整恢复（同阶段断点续训）
- 不匹配 → 仅加载模型权重，optimizer 重新初始化（跨阶段结构变化）

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
| `use_semantic` | `false` | 语义模块开关（Stage 2 开启） |
| `use_sr_condition` | `false` | SR 条件分支开关（Stage 2 开启） |
| `use_confidence_gate` | `false` | 置信度门控开关（Stage 3 开启） |
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
| `sr_fixed` | `true` | SR Prior 是否冻结 |
| `precision` | `"bf16-mixed"` | 混合精度训练 |

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
| `train/D_tex_conf` | D_tex 置信度均值 |
| `val/psnr` | 验证 PSNR |
| `val/lpips` | 验证 LPIPS |

## 项目结构

```
RefRWKV/
├── configs/
│   ├── stage1_baseline.yaml          # Stage 1：基础扩散
│   ├── stage2_semantic.yaml          # Stage 2：语义 + SR 条件
│   ├── stage3_texture.yaml           # Stage 3：纹理精调
│   ├── stage4_gan.yaml              # Stage 4：GAN 精调
│   └── sr_prior.yaml                # SR Prior 独立训练
├── models/
│   ├── RefSRWKV.py                    # SR Prior（RWKV 超分模型）
│   ├── cuda/                          # RWKV WKV CUDA 算子
│   └── RefDiffRWKV/
│       ├── sd2_ref_generator.py       # 扩散生成器
│       ├── sd2_ref_discriminator.py   # 双判别器
│       ├── sd2_ref_gan_system.py      # GAN 训练系统
│       ├── sd2_ref_adapter.py         # 参考图 Adapter
│       ├── globalsemanticmodule.py    # DINOv2 + RWKV 语义模块
│       └── spaced_sampler.py          # 采样器
├── scripts/
│   ├── train_sd2_gan.py              # 训练脚本
│   └── verify_sr_ckpt.py             # SR 权重兼容性验证
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
| 跨阶段 optimizer 预检测 | 避免参数数量不匹配时双重加载 checkpoint 导致 OOM |
| bf16-mixed 混合精度 | 降低显存占用，配合 `.float()` 转换兼容 Numpy/IQA |

## 已知限制

- 仅支持 10 倍超分（48×48 → 480×480）
- 单卡训练（RTX 4090 24GB）
- 评估指标以 Y-channel PSNR/SSIM 为主，遥感专用指标（SAM/ERGAS）待集成
- CUDA WKV 算子需要编译，不支持纯 CPU 推理
- `semantic_pyramid` 权重每次恢复都跳过（WKV 公式变更遗留），需改为条件跳过
- `SelfSimTransfer` 全局 affinity 为 O(N²)，高分辨率下显存风险（当前已关闭）
- DataLoader 不支持断点续训恢复（epoch 中间断开后重新遍历）

---

## 改进路线图（Roadmap）

> 以下按优先级排列。标注 ⏳ 的为当前阶段暂不修改，等训练跑完后再处理。

### 🔴 P0：实验设计（论文核心）

#### 1. 参考图鲁棒性实验（Reviewer 必问）

- [ ] 训练时随机注入 10%~20% 错误参考图（`ref_drop_prob`）
- [ ] 验证集构造三种场景：正确参考 / 错位参考 / 不相关参考
- [ ] 增加约束：错误参考图不应使结果明显差于无参考基线
- [ ] 验证模型在不可信参考下退化为单图超分的能力

#### 2. 最小实验矩阵（消融实验）

| 实验 | RWKV | SD2 prior | Reference | Semantic | GAN |
|---|---:|---:|---:|---:|---:|
| A（Baseline） | ✓ |  |  |  |  |
| B | ✓ | ✓ |  |  |  |
| C | ✓ | ✓ | ✓ |  |  |
| D（当前） | ✓ | ✓ | ✓ | ✓ |  |
| E（完整） | ✓ | ✓ | ✓ | ✓ | ✓ |

每组实验额外测试：无参考 / 正确参考 / 错误参考 / 合成退化 / 真实退化。

#### 3. 评估指标扩展

- [ ] 保存三类 checkpoint：`best_fidelity`（PSNR/SSIM）/ `best_perceptual`（LPIPS/DISTS）/ `best_balanced`
- [ ] 综合分数固定公式：`score = norm_psnr + norm_ssim - norm_lpips - norm_dists`
- [ ] 集成遥感专用指标：SAM、ERGAS、边缘保持、下采样一致性
- [ ] 可选：地物分割/检测下游任务性能评估

---

### 🟡 P1：方法优化（Stage 3/4 期间可加）

#### 4. 10 倍超分分阶段结构监督

- [ ] 在 RWKV 中间层增加 5× 中间尺度监督出口
- [ ] 中间尺度 L1/Charbonnier loss
- [ ] 梯度/边缘损失（Gradient Loss）
- [ ] 最终尺度：重建 + 感知 + 扩散 + 对抗

#### 5. 对抗学习聚焦高频

- [ ] D_tex 输入改为高通滤波后的图像（Laplacian / Sobel）
- [ ] D_sem 使用较大感受野、较低分辨率
- [ ] D_tex 使用局部 patch、高频输入
- [ ] GAN 权重线性 warmup：`lambda_gan(step): 0 → linear → target`

#### 6. LR Cycle Consistency

- [ ] 将 SR 结果按已知退化降采样，约束接近输入 LR
- [ ] `loss_cycle = distance(degrade(sr), lr)`
- [ ] 限制生成结果偏离输入中的可观测结构

#### 7. 训练退化模型增强

- [ ] 混合退化：bicubic / area / Lanczos / Gaussian blur / motion blur / sensor MTF
- [ ] 噪声：Gaussian / Poisson
- [ ] 压缩：JPEG
- [ ] 遥感特有：轻微配准误差、大气/雾霾、对比度变化、波段偏移
- [ ] 记录每个样本的退化参数，便于失败分析

---

### 🟢 P2：架构改进（V2 版本）

#### 8. 参考融合改为 Confidence-Gated Transfer

```python
fused = sr_feature + confidence * transfer(reference_feature)
```

- [ ] confidence 综合：几何对应质量 + 语义一致性 + 局部纹理一致性
- [ ] 参考 Dropout：训练时随机丢弃参考，让模型学会退化
- [ ] 推理时可视化 confidence map，分析哪些区域被参考引导

#### 9. Self-Similarity 改为分层候选检索

- [ ] 低分辨率特征做粗粒度全局检索（top-k 候选区域）
- [ ] 高分辨率特征只在候选邻域内做局部精细匹配
- [ ] 复杂度从 O(N²) 降至 O(NK)
- [ ] 可选：Windowed Attention / Top-K Sparse Affinity / Coarse-to-Fine

#### 10. 明确模块职责分工

```
RWKV         → 长程空间依赖、全局布局、道路连续性
SD2 latent   → 自然高频统计和生成先验
Reference    → 只提供可验证的对应纹理（confidence-gated）
GAN          → 只约束最终高频/纹理频段
```

- [ ] 生成部分预测高频残差：`sr = structural_base + high_frequency_residual`
- [ ] 降低扩散先验覆盖 LR 几何信息的概率

---

### 🔵 P3：工程优化（训练跑完后重构）

#### 11. 阶段状态机显式化

- [ ] 定义 `TrainingStage` dataclass（name, trainable_modules, optimizer_layout）
- [ ] 分离 `resume_from`（同阶段严格恢复）和 `finetune_from`（跨阶段仅加载权重）
- [ ] Checkpoint 保存元数据：`schema_version / stage_name / config_hash / optimizer_signature`
- [ ] 恢复前逐项校验，不匹配时输出具体差异

#### 12. D warmup 配置化

- [ ] 将硬编码的 `global_step < 3000` 改为 `self.d_warmup_steps`
- [ ] 配置文件中增加 `d_warmup_steps` 参数
- [ ] 基于有效 optimizer step 而非 Trainer global_step

#### 13. semantic_pyramid 跳过逻辑改为条件跳过

- [ ] 当前：无条件跳过所有 `semantic_pyramid.*` 权重
- [ ] 改为：仅当 checkpoint 中的 shape 与当前模型不匹配时才跳过
- [ ] 同阶段断点续训时正常加载，保留训练进度

#### 14. OOM 重试路径统一

- [ ] OOM 重建后重新执行 optimizer 兼容性预检
- [ ] 将 checkpoint 恢复决策集中到单一函数
- [ ] 正常路径和 OOM 重试路径使用同一套决策逻辑

#### 15. 其他工程修复

- [ ] Sampler 增加 `steps < 2` 显式校验
- [ ] CUDA WKV 扩展延迟加载（支持 CPU 导入）
- [ ] LPIPS 仅在 `lambda_lpips > 0` 时初始化
- [ ] 移除采样过程中的 `print()` 语句，改为可配置 logger
- [ ] DataLoader 实现 `state_dict` / `load_state_dict` 支持断点续训
- [ ] 配置启动时统一校验（尺寸、指标、依赖、显存估算）

---

### 实施顺序建议

```
当前：Stage 2 训练中（不要改代码）
  ↓
Stage 2 跑完后：
  1. 跑 Baseline 对比实验（Bicubic / SRCNN / ESRGAN / SwinIR）
  2. 构造错误参考图测试集
  3. 启动 Stage 3
  ↓
Stage 3/4 期间：
  4. 加入 D_tex 高频输入（P1-5）
  5. 加入 LR Cycle Consistency（P1-6）
  6. 加入退化模型增强（P1-7）
  ↓
全部 Stage 跑完后：
  7. 完成消融实验矩阵（P0-2）
  8. 参考图鲁棒性实验（P0-1）
  9. 评估指标扩展（P0-3）
  ↓
论文投稿后：
  10. 工程重构（P3 全部）
  11. 架构改进 V2（P2 全部）
```
```

---

这份 README 现在包含了完整的项目文档 + 改进路线图。你可以直接替换原来的 README 文件。路线图按优先级排列，方便你后续逐步推进。