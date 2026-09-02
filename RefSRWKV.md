# RefSRWKV

RefSRWKV 是用于遥感图像超分辨率的 RGB 重建先验网络。它以窗口化双向 RWKV 为主干，在 U-Net 的四个尺度融合 LR 与参考特征，并以残差方式恢复 HR 图像。训练和推理张量均使用 [-1, 1] 值域。

## 输入与尺寸

~~~text
lr:   [B, C_in, H_lr, W_lr]
ref:  [B, C_in, H_hr, W_hr]
out:  [B, C_out, H_hr, W_hr]
~~~

- ref_channels 必须等于 inp_channels，因为参考图会先与 LR 的 bicubic 上采样结果做逐样本颜色统计匹配。
- dim 必须是 16 的倍数，满足 CUDA Bi-WKV 算子的通道约束。
- 配置中的 hr_size 是训练 crop 与 checkpoint 的尺寸契约，必须可被 32 整除。
- 推理时网络从输入 ref 的实际尺寸推导内部网格，不会把更大的测试图缩回训练 crop。运行时 ref 的高和宽也必须可被 32 整除。
- scale 描述数据 loader 中 LR/HR crop 的对应关系；模型输出的空间尺寸始终跟随 ref。

参考图先用 PixelUnshuffle(4) 进入 HR/4 网格，经过三级下采样与对称解码后，再由两级 PixelShuffle(2) 重建 HR 残差：

~~~text
out = clamp(bicubic(lr, ref.size) + residual, -1, 1)
~~~

最终输出卷积零初始化，因此训练开始时的预测就是 bicubic 基线。

## 分层窗口 RWKV

每个 Block 只执行一次窗口内 Bi-WKV。offsets 定义同一 stage 内各 block 轮流使用的窗口原点；例如 [0, 4] 表示第 0 个 block 不移位、第 1 个 block 移位 4 像素、第 2 个 block 再回到不移位。它不是在单个 block 内额外计算两次。

默认训练配置位于 configs/sr_prior_base.yaml，采用每个 stage 独立重新计相位的对称方案：

| Stage | AID x4 特征尺寸 | window / offsets |
| --- | ---: | --- |
| enc1、dec1、refine | 64 x 64 | 8 / [0, 4] |
| enc2、dec2 | 32 x 32 | 8 / [0, 4] |
| enc3、dec3 | 16 x 16 | 4 / [0, 2] |
| latent | 8 x 8 | 4 / [0, 2] |

窗口边界以零填充移动，不发生循环拼接。窗口大小和偏移量不引入可学习参数，也不改变 RWKV、卷积或融合层的张量形状。因此，在 dim、hidden_rate、通道数、block 数和其余网络结构相同的前提下，改变窗口配置可以通过 --load_weights 迁移权重。

没有 windows 字段时，模型保持全局相位的兼容模式：window_size: 8、shift_size: 3、shift_cycle: 3 对应 8 / [0, 3, 6]。

## 参考模式

data.reference_mode 控制进入模型的参考图：

| 模式 | 行为 | 使用数据集 |
| --- | --- | --- |
| lr_up | 在运行时把当前 LR bicubic 上采样到 HR 尺寸，忽略存储的 Ref 内容 | AID、UC Merced 的 SISR |
| paired | 使用 PNG 中同名的真实配对 Ref | HRMS-SCD、Real-RefRSSRD |

model.ref_drop_prob 仅在训练阶段生效，并按样本将配对参考替换为该样本的 LR 上采样图。AID 的 aid_x4_l1 使用 lr_up 与 ref_drop_prob: 0.0，因此训练、验证和测试始终是相同的单图超分辨率条件。

GatedFusion 在四个尺度融合 LR 与参考特征。它以余弦相似度构成置信度，再通过可学习门控控制参考信息：

~~~text
lr_feature + gate(fused) * confidence * fused
~~~

## CUDA 环境

空间 RWKV 只使用 models/cuda/bi_wkv.cpp 和 models/cuda/bi_wkv_kernel.cu 提供的 CUDA Bi-WKV 实现。首次前向传播时，PyTorch 通过 torch.utils.cpp_extension.load 编译并加载扩展；不提供 PyTorch 等价后端。

已验证环境：Conda rwkv7、PyTorch 2.10.0+cu128、CUDA 12.8、RTX 5060 Ti。

主要依赖：

~~~text
torch
einops
pytorch-lightning
torchvision
Pillow
PyYAML
pyiqa  # 可选；不可用时使用内置 SSIM
~~~

## 数据与配置

所有数据集使用 PNG 三元组：

~~~text
<data_root>/<split>/LR/*.png
<data_root>/<split>/HR/*.png
<data_root>/<split>/Ref/*.png
~~~

RefPNGDataset 从 LR 坐标采样 crop，再按 scale 映射到 HR 与 Ref，以保持像素对齐。训练 crop 随机采样；验证和测试 crop 使用样本索引与固定种子，保证每次验证覆盖相同区域。

配置分为三层：

~~~text
configs/sr_prior_base.yaml       # 网络、窗口、优化器和通用训练默认值
configs/datasets/*.yaml          # 数据集事实、根目录与存储尺寸
configs/runs/*.yaml              # 训练 HR crop、倍率、loss 与实验差异
~~~

run 配置通过 base: 继承数据集配置，脚本会自动展开为 data.root、data.patch_size、data.scale 和 model.scale。数据集信息见以下说明：

- [AID](data/remote_sensing/prepared/AID/介绍.md)
- [UC Merced](data/remote_sensing/prepared/UC_Merced/介绍.md)
- [RefSR-HRMS](RefSR_data/HRMS_SCD/RefSR-HRMS.md)
- [Real-RefRSSRD](RefSR_data/ALL_2/Real-RefRSSRD.md)

## 训练

| 配置 | HR/LR train crop | 参考模式 | batch | 训练终点 |
| --- | --- | --- | ---: | --- |
| configs/runs/aid_x4_l1.yaml | 256 / 64, x4 | lr_up | 32 | L1，50,000 optimizer steps |
| configs/runs/hrms_scd_x4.yaml | 512 / 128, x4 | paired | 4 | 50,000 optimizer steps |
| configs/runs/ucmerced_x4.yaml | 256 / 64, x4 | lr_up | 4 | 50 epochs |
| configs/runs/real_refrssrd_x10.yaml | 480 / 48, x10 | paired | 4 | 200 epochs |

先从头训练 AID：

~~~bash
conda run -n rwkv7 python scripts/train_sr_prior.py \
  --config configs/runs/aid_x4_l1.yaml
~~~

首次运行会创建 checkpoints/refrwkv_sr_aid_x4_l1/。同一命令再次执行会自动检测该目录的兼容 last.ckpt 并恢复完整训练状态。需要创建一条全新的 AID 实验时，覆盖 run 名称：

~~~bash
conda run -n rwkv7 python scripts/train_sr_prior.py \
  --config configs/runs/aid_x4_l1.yaml \
  --overrides run.name=aid_x4_l1_run2
~~~

完成 AID 后，将其 EMA 权重迁移到 HRMS-SCD：

~~~bash
conda run -n rwkv7 python scripts/train_sr_prior.py \
  --config configs/runs/hrms_scd_x4.yaml \
  --load_weights checkpoints/refrwkv_sr_aid_x4_l1/last.ckpt
~~~

--load_weights 优先加载 checkpoint 中的 EMA 参数，只迁移同名且形状匹配的模型权重；Adam、学习率调度器、global step 和 EMA 计数从当前 run 重新开始。--resume 只用于数据、窗口、模型和训练语义均相同的完整续训。

若 AID 的 batch 32 在实际训练中显存不足，保持等效 batch 32：

~~~bash
conda run -n rwkv7 python scripts/train_sr_prior.py \
  --config configs/runs/aid_x4_l1.yaml \
  --overrides data.batch_size=16 train.accumulate_grad_batches=2
~~~

## 学习率与验证

默认使用 ReduceLROnPlateau。调度器在每次验证结束后读取聚合的 val_loss，不是只读取 epoch 的最后一次验证：

- lr_patience: 2：连续两次无有效改善后，第三次无效验证触发降学习率。
- lr_threshold: 1.0e-4：只有 val_loss 至少下降该绝对值才视为改善。
- val_check_interval 决定验证频率，因此 patience 的单位是验证次数。
- AID 使用确定性的 300 张验证子集与 val_check_interval: 0.5，使每次 plateau 判断可比较且成本受控。

EMA 只在真实 optimizer step 后更新，并在验证、测试和 checkpoint 热启动时使用。

## 评测

scripts/eval_four_settings.py 支持四种输入条件：

- bicubic：插值基线。
- sisr_ref：当前 LR 的 bicubic 上采样，适用于 AID 与 UC Merced。
- dataset_ref：数据集中的真实配对 Ref，适用于 HRMS-SCD 与 Real-RefRSSRD。
- perfect_ref：将 HR 当作参考，仅用于诊断上限。

训练完成后评测 AID 的完整 test split：

~~~bash
conda run -n rwkv7 python scripts/eval_four_settings.py \
  --ckpt checkpoints/refrwkv_sr_aid_x4_l1/last.ckpt \
  --data data/remote_sensing/prepared/AID \
  --splits test \
  --settings bicubic sisr_ref \
  --batch-size 1
~~~

迁移训练后的 HRMS-SCD 使用两个测试 split：

~~~bash
conda run -n rwkv7 python scripts/eval_four_settings.py \
  --ckpt checkpoints/refrwkv_sr_hrms_scd_x4/last.ckpt \
  --data RefSR_data/HRMS_SCD \
  --splits test_easy test_hard \
  --settings bicubic dataset_ref sisr_ref \
  --batch-size 1
~~~

评测脚本优先从 checkpoint 签名读取 scale、训练 hr_size 和窗口配置；不需要手工重复这些参数。

## 扩散阶段集成

scripts/train_sd2_gan.py 从 configs/base.yaml 的 model.sr 构建 SR prior，并传递相同的窗口配置。model.sr.ckpt_path 默认是 null，避免扩散阶段意外加载不属于当前实验的 SR 权重；开始扩散训练前应显式设置为相应 SR checkpoint。
