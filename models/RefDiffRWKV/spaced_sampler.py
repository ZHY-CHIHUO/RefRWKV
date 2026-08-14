"""
SpacedSampler: IDDPM 加速采样引擎 / IDDPM Accelerated Sampling Engine

基于 IDDPM (Improved Denoising Diffusion Probabilistic Models) 的间隔采样方案，
将原始的 1000 步扩散过程压缩到用户指定的步数（如 50 步），在不显著损失质量的前提下
大幅加速推理。
Based on IDDPM's spaced sampling schedule, compresses the original 1000-step
diffusion process into a user-specified number of steps (e.g., 50) for much
faster inference with negligible quality loss.

核心原理 / Core Principle:
    从 1000 步中均匀抽取 N 个时间步，重新计算每步对应的 β 和 α 参数，
    使得 q(x_{t-1}|x_t, x_0) 的边际分布与原始过程一致。
    Uniformly subsample N timesteps from 1000, then recompute β and α
    parameters per step so the marginal q(x_{t-1}|x_t, x_0) stays
    consistent with the original process.

参考 / Reference:
    https://arxiv.org/pdf/2102.09672.pdf
    https://github.com/openai/guided-diffusion
"""

from typing import Optional, Tuple, Dict
import torch
import numpy as np
from tqdm import tqdm
from .cond_fn import Guidance
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# 噪声调度函数 / Noise Schedule Function
# ============================================================================

def make_beta_schedule(
    schedule: str,
    n_timestep: int,
    linear_start: float = 1e-4,
    linear_end: float = 2e-2,
) -> np.ndarray:
    """
    生成扩散过程前向加噪的 β 序列 / Generate β sequence for forward diffusion.

    扩散模型的前向过程: x_t = √(ᾱ_t) · x_0 + √(1-ᾱ_t) · ε
    其中 ᾱ_t = ∏(1 - β_s)，β 越大则这一步加入的噪声越多。
    Forward diffusion: x_t = √(ᾱ_t) · x_0 + √(1-ᾱ_t) · ε
    where ᾱ_t = ∏(1 - β_s). Larger β → more noise added at this step.

    两种调度方案 / Two schedule types:
        - "linear":      β 线性增长 / β grows linearly
        - "scaled_linear": β 在 log 空间线性增长（原版 IDDPM 方案）
                           β grows linearly in log-space (original IDDPM)

    Args:
        schedule (str):      调度类型 / Schedule type ("linear" or "scaled_linear")
        n_timestep (int):    总时间步数（通常 1000）/ Total timesteps (usually 1000)
        linear_start (float): 第一步的 β 值（≈1e-4）/ Starting β at t=1
        linear_end (float):   最后一步的 β 值（≈2e-2）/ Ending β at t=T

    Returns:
        np.ndarray: β 序列，形状 (n_timestep,) / β sequence, shape (n_timestep,)
    """
    if schedule == "linear":
        # 线性增长: β_t 在 [start, end] 之间均匀分布
        # Linear increase: β_t evenly spaced between [start, end]
        betas = (
            torch.linspace(
                linear_start ** 0.5, linear_end ** 0.5,
                n_timestep, dtype=torch.float64
            ) ** 2
        )
        return betas.numpy()
    elif schedule == "scaled_linear":
        # 缩放线性: 同上但直接用 numpy（原版 guided-diffusion 风格）
        # Scaled linear: same as above but using numpy (original style)
        betas = np.linspace(
            linear_start ** 0.5, linear_end ** 0.5,
            n_timestep, dtype=np.float64
        ) ** 2
        return betas
    raise NotImplementedError(f"Unknown schedule: {schedule}")


# ============================================================================
# 时间步子采样函数 / Timestep Subsampling Function
# ============================================================================

def space_timesteps(
    num_timesteps: int,
    section_counts
) -> set:
    """
    从原始时间步中等距抽取子集 / Subsample timesteps evenly from the original set.

    将 1000 步分成若干段，每段内等距抽取指定数量的时间步，实现加速采样。
    Divide 1000 steps into sections, then evenly sample a specified number of
    timesteps within each section for accelerated sampling.

    DDIM 模式 / DDIM mode:
        "ddim50": 从 {0, 20, 40, ..., 980} 中抽取 50 个步
        从 1000 步中以整数步长等距抽取 50 步
        "ddim50": sample 50 steps evenly from {0, 20, 40, ..., 980}
        Take every 1000/50 = 20 steps

    分段模式 / Multi-section mode:
        "10,15,20": 前 333 步取 10 个，中间 333 步取 15 个，后 334 步取 20 个
        早期噪声大可用较少步，后期细节多需更多步
        "10,15,20": 10 from first 333, 15 from middle 333, 20 from last 334
        Early steps (more noise) need fewer steps; later steps need more detail

    Args:
        num_timesteps (int): 原始总步数（1000）/ Original total steps
        section_counts:      ddim 字符串或分段计数字符串
                             ddim string or comma-separated section counts

    Returns:
        set: 被选中的时间步索引集合 / Set of selected timestep indices
    """
    # DDIM 模式：等距抽取 / DDIM mode: uniform stride
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim"):])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        # 分段模式 / Multi-section mode
        section_counts = [int(x) for x in section_counts.split(",")]

    # 将总时间步均分给各段 / Divide total steps evenly among sections
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            # 段内等距步长 / Even stride within section
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


# ============================================================================
# 工具函数 / Utility Function
# ============================================================================

def _extract_into_tensor(
    arr: np.ndarray,
    timesteps: torch.Tensor,
    broadcast_shape: tuple
) -> torch.Tensor:
    """
    从 numpy 数组中按时间步索引提取值，并广播到目标形状 / Extract values
    from a numpy array by timestep index and broadcast to target shape.

    用于从预计算的 α/β 表格中按当前时间步 t 取对应的系数，然后
    广播到与 latent tensor 相同的形状（B, C, H, W）。
    Used to fetch precomputed α/β coefficients by current timestep t,
    then broadcast to match the latent tensor shape (B, C, H, W).

    Args:
        arr (np.ndarray):   预计算的参数表 / Precomputed parameter table
        timesteps (Tensor):  当前时间步索引 / Current timestep indices  (N,)
        broadcast_shape (tuple): 目标形状 / Target shape for broadcasting

    Returns:
        Tensor: 广播后的系数张量 / Broadcast coefficient tensor
    """
    try:
        # 默认 float64 → 转 float32 / Default float64 → convert to float32
        res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    except:
        # MPS 设备不兼容 float64 时的回退 / Fallback for MPS float64 incompatibility
        res = (
            torch.from_numpy(arr.astype(np.float32))
            .to(device=timesteps.device)[timesteps]
            .float()
        )
    # 逐维扩展到目标形状 / Expand dims to match target shape
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


# ============================================================================
# SpacedSampler: IDDPM 间隔采样器 / IDDPM Spaced Sampler
# ============================================================================

class SpacedSampler:
    """
    实现 IDDPM 论文提出的间隔采样方案，用于 ControlLDM 的推理。
    Implementation of IDDPM's spaced sampling schedule for ControlLDM inference.

    使用方式 / Usage:
        sampler = SpacedSampler(model)
        hr_image = sampler.sample(
            steps=50,                    # 采样步数 / sampling steps
            shape=(1, 4, 60, 60),       # latent 形状 / latent shape
            cond=condition_dict,         # 条件字典（含 ref 特征）/ condition dict
        )

    两种方差策略 / Two variance strategies:
        - "fixed_small": 用后验方差 (较小，稳定) / Use posterior variance (smaller, stable)
        - "fixed_large": 用 β_t 作为方差 (较大，多样性高) / Use β_t as variance (larger, more diverse)
    """

    def __init__(
        self,
        model,                        # ControlLDM 实例 / ControlLDM instance
        schedule: str = "linear",     # 噪声调度类型 / noise schedule type
        var_type: str = "fixed_small" # 方差类型 / variance type
    ) -> "SpacedSampler":
        """
        Args:
            model: ControlLDM 或类似模型，需要提供以下接口:
                   ControlLDM or similar model, must provide:
                   - model.num_timesteps: 原始扩散步数 / original diffusion steps
                   - model.linear_start / linear_end: β 边界 / β bounds
                   - model.apply_model(x, t, cond): 单步噪声预测 / single-step noise pred
                   - model.decode_first_stage(latent): VAE 解码到像素 / VAE decode to pixel
            schedule: "linear" 或 "scaled_linear"
            var_type: "fixed_small" 或 "fixed_large"
        """
        self.model = model
        self.original_num_steps = model.num_timesteps  # 通常 1000 / usually 1000
        self.schedule = schedule
        self.var_type = var_type

    # ========================================================================
    # 构建采样参数 / Build Sampling Parameters
    # ========================================================================

    def make_schedule(self, num_steps: int) -> None:
        """
        根据目标采样步数计算所有必需的扩散参数 / Precompute all diffusion
        parameters for the target number of sampling steps.

        计算流程 / Computation flow:
            1. 生成原始 1000 步的 β 序列 / Generate original 1000-step β sequence
            2. 计算原始 ᾱ 累积乘积 / Compute original ᾱ cumulative product
            3. 子采样 N 个时间步 / Subsample N timesteps
            4. 重新计算每步对应的 β, α, ᾱ  / Recompute per-step β, α, ᾱ
            5. 预计算后验分布 q(x_{t-1}|x_t, x_0) 的均值和方差系数
               Precompute posterior mean and variance coefficients

        Args:
            num_steps (int): 目标采样步数（如 50）/ Target sampling steps (e.g., 50)
        """
        if num_steps < 2 or num_steps > self.original_num_steps:
            raise ValueError(
                f"num_steps({num_steps}) 必须在 [2, {self.original_num_steps}] 范围内"
            )

        # 步骤 1-2: 原始扩散参数 / Step 1-2: Original diffusion parameters
        original_betas = make_beta_schedule(
            self.schedule,
            self.original_num_steps,
            linear_start=self.model.linear_start,
            linear_end=self.model.linear_end,
        )
        original_alphas = 1.0 - original_betas
        original_alphas_cumprod = np.cumprod(original_alphas, axis=0)

        # 步骤 3: 子采样时间步 / Step 3: Subsample timesteps
        used_timesteps = space_timesteps(self.original_num_steps, str(num_steps))
        logger.debug(
            "timesteps used in spaced sampler: %s", sorted(list(used_timesteps))
        )

        # 步骤 4: 重新计算每步 beta / Step 4: Recompute per-step beta
        # 保持边际分布一致: q(x_{S_t}|x_0) 与原始过程相同
        # Keep marginal consistent: q(x_{S_t}|x_0) matches original process
        betas = []
        last_alpha_cumprod = 1.0
        for i, alpha_cumprod in enumerate(original_alphas_cumprod):
            if i in used_timesteps:
                betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
        if len(betas) != num_steps:
            raise ValueError(
                f"采样子采样产生 {len(betas)} 步，期望 {num_steps} 步"
            )
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas

        # 采样步的实际索引 / Actual timestep indices used
        self.timesteps = np.array(sorted(list(used_timesteps)), dtype=np.int32)
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)

        # ---- 前向过程参数 / Forward Process Parameters ----
        # q(x_t | x_0) 的参数 / Parameters for q(x_t | x_0)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        # 从 x_t 和 ε 反推 x_0 的系数 / Coefficients to recover x_0 from x_t and ε
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # ---- 后验分布参数 / Posterior Distribution Parameters ----
        # q(x_{t-1} | x_t, x_0) 的方差 / Variance of q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # 对数方差（裁剪防止除零）/ Log variance (clipped to avoid division by zero)
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        # 后验均值的两个系数 / Two coefficients for posterior mean
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    # ========================================================================
    # 扩散过程 / Diffusion Process
    # ========================================================================

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        前向加噪: 对干净图像 x_0 加噪得到 x_t / Forward diffusion: add noise
        to clean image x_0 to get noisy image x_t.

        公式 / Formula: x_t = √(ᾱ_t) · x_0 + √(1-ᾱ_t) · ε

        Args:
            x_start (Tensor): 干净图像 (N,C,H,W) / Clean images
            t (Tensor):       时间步索引 (N,) / Timestep indices
            noise (Tensor):   可选，指定噪声 / Optional, specify noise

        Returns:
            Tensor: 加噪后的图像 / Noisy images x_t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(
        self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算后验分布 q(x_{t-1} | x_t, x_0) 的均值和方差 /
        Compute the mean and variance of posterior q(x_{t-1} | x_t, x_0).

        在已知 x_0 和当前 x_t 的条件下，x_{t-1} 服从高斯分布。
        此方法给出该高斯分布的均值和方差，用于采样 x_{t-1}。
        Given clean x_0 and current x_t, x_{t-1} follows a Gaussian.
        Returns its mean and variance for sampling x_{t-1}.

        Args:
            x_start (Tensor): 预测的干净图像 / Predicted clean image x_0
            x_t (Tensor):     当前带噪图像 / Current noisy image x_t
            t (Tensor):       后验分布的时间步索引 / Posterior timestep index

        Returns:
            (mean, variance, log_variance_clipped): 后验分布的三个参数
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(
            self.posterior_variance, t, x_t.shape
        )
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    # ========================================================================
    # 预测函数 / Prediction Functions
    # ========================================================================

    def _predict_xstart_from_eps(
        self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        """
        从预测的噪声 ε 反推干净图像 x_0 / Recover clean image x_0 from predicted
        noise ε using:
            x_0 = x_t / √(ᾱ_t) - ε · √(1/ᾱ_t - 1)

        Args:
            x_t (Tensor): 当前带噪图像 / Current noisy image
            t (Tensor):   时间步索引 / Timestep index
            eps (Tensor):  模型预测的噪声 / Model-predicted noise

        Returns:
            Tensor: 预测的干净图像 / Predicted clean image x_0
        """
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def predict_noise(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        cfg_scale: float,
        uncond: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        """
        预测当前时间步的噪声 / Predict noise at current timestep.

        支持无分类器引导 (CFG) / Supports classifier-free guidance:
            ε = ε_uncond + cfg_scale · (ε_cond - ε_uncond)

        Args:
            x (Tensor):        当前带噪 latent / Current noisy latent
            t (Tensor):        时间步 / Timestep
            cond (Dict):       有条件的条件字典 / Conditional condition dict
            cfg_scale (float):  CFG 引导强度 (>1 增强条件控制) / CFG strength
            uncond (Dict):     无条件的条件字典 / Unconditional condition dict

        Returns:
            Tensor: 预测的噪声 ε / Predicted noise
        """
        if uncond is None or cfg_scale == 1.0:
            # 不加 CFG / No CFG
            model_output = self.model.apply_model(x, t, cond)
        else:
            # 无分类器引导 / Classifier-free guidance
            model_cond = self.model.apply_model(x, t, cond)
            model_uncond = self.model.apply_model(x, t, uncond)
            model_output = model_uncond + cfg_scale * (model_cond - model_uncond)

        # v-参数化需要转换 / v-parameterization needs conversion
        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output
        return e_t

    # ========================================================================
    # 隐式引导 (MSE Guidance) / Latent Image Guidance
    # ========================================================================

    def apply_cond_fn(
        self,
        x: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        t: torch.Tensor,
        index: torch.Tensor,
        cond_fn: Guidance,
        cfg_scale: float,
        uncond: Optional[Dict[str, torch.Tensor]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        应用隐式引导：用目标图像约束 pred_x0 的方向 /
        Apply latent image guidance: steer pred_x0 towards target image.

        在去噪过程中，每一步预测出干净图像 pred_x0 后，计算它与
        目标图像（如 rwkvSR 输出）之间的 MSE loss，用梯度修正 pred_x0
        使其更接近目标，从而保留结构保真度。
        During denoising, after predicting clean image pred_x0 each step,
        compute MSE loss against target (e.g., rwkvSR output), then
        gradient-correct pred_x0 towards the target for structural fidelity.

        Args:
            x (Tensor):        当前 noisy latent / Current noisy latent
            cond (Dict):       条件字典 / Condition dict
            t (Tensor):        时间步 / Timestep
            index (Tensor):    采样索引 / Sampling index
            cond_fn (Guidance): 引导实例 (如 MSEGuidance) / Guidance instance
            cfg_scale (float):  CFG 强度 / CFG strength
            uncond (Dict):     无条件条件字典 / Unconditional condition dict

        Returns:
            (model_mean, pred_x0): 修正后的后验均值和 pred_x0
        """
        device = x.device
        t_now = int(t[0].item()) + 1

        # 步骤1: 预测噪声和 x_0 / Step 1: Predict noise and x_0
        e_t = self.predict_noise(x, t, cond, cfg_scale, uncond)
        pred_x0: torch.Tensor = self._predict_xstart_from_eps(x_t=x, t=index, eps=e_t)
        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_x0, x_t=x, t=index
        )

        # 步骤2: 多次梯度修正 / Step 2: Multiple gradient correction rounds
        for _ in range(cond_fn.repeat):
            target, pred = None, None
            if cond_fn.space == "latent":
                # 在 latent 空间计算 loss / Compute loss in latent space
                target = self.model.get_first_stage_encoding(
                    self.model.encode_first_stage(cond_fn.target.to(device))
                )
                pred = pred_x0
            elif cond_fn.space == "rgb":
                # 在 RGB 空间计算 loss，需要追踪梯度回 latent 空间
                # Compute loss in RGB space, trace gradients back to latent
                with torch.enable_grad():
                    pred_x0.requires_grad_(True)
                    target = cond_fn.target.to(device)
                    pred = self.model.decode_first_stage_with_grad(pred_x0)
            else:
                raise NotImplementedError(cond_fn.space)

            # 计算梯度修正量 / Compute gradient correction
            delta_pred = cond_fn(target, pred, t_now)

            if delta_pred is not None:
                if cond_fn.space == "rgb":
                    # RGB 空间: 链式法则回传 / RGB space: chain rule backprop
                    pred.backward(delta_pred)
                    delta_pred_x0 = pred_x0.grad
                    pred_x0 += delta_pred_x0
                    model_mean += 0.5 * delta_pred_x0  # 0.5 是经验系数 / heuristic
                    pred_x0.grad.zero_()
                else:
                    # Latent 空间: 直接加 / Latent space: direct addition
                    delta_pred_x0 = delta_pred
                    pred_x0 += delta_pred_x0
                    model_mean += 0.5 * delta_pred_x0
            else:
                # delta_pred 为 None 表示停止引导 / None means stop guidance
                break

        return model_mean.detach().clone(), pred_x0.detach().clone()

    # ========================================================================
    # 单步采样 / Single Sampling Step
    # ========================================================================

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        t: torch.Tensor,
        index: torch.Tensor,
        cfg_scale: float,
        uncond: Optional[Dict[str, torch.Tensor]],
        cond_fn: Optional[Guidance],
    ) -> torch.Tensor:
        """
        单步去噪采样: x_t → x_{t-1} / Single denoising step: x_t → x_{t-1}.

        实现 DDPM/DDIM 采样公式 / Implements DDPM/DDIM sampling formula:
            x_{t-1} = μ_θ(x_t, t) + σ_t · z    (z ~ N(0,I))
        其中 μ_θ 由预测的 x_0 通过后验分布计算得出。
        where μ_θ is computed from predicted x_0 via the posterior.

        Args:
            x (Tensor):        当前 noisy latent (B,C,H,W)
            cond (Dict):       条件字典 / Condition dict
            t (Tensor):        时间步 / Timestep (B,)
            index (Tensor):    采样索引 / Sampling index (B,)
            cfg_scale (float):  CFG 引导强度 / CFG scale
            uncond (Dict):     无条件条件字典 / Unconditional cond dict
            cond_fn (Guidance): 可选的隐式引导 / Optional guidance function

        Returns:
            Tensor: x_{t-1}，上一步的 noisy latent / Previous noisy latent
        """
        # ---- 步骤1: 确定方差 / Step 1: Determine variance ----
        model_variance = {
            # fixed_large: 用 β_t 作方差，多样性更高 / use β_t, more diverse
            "fixed_large": np.append(self.posterior_variance[1], self.betas[1:]),
            # fixed_small: 用后验方差，更稳定 / use posterior variance, more stable
            "fixed_small": self.posterior_variance,
        }[self.var_type]
        model_variance = _extract_into_tensor(model_variance, index, x.shape)

        # ---- 步骤2: 计算后验均值 / Step 2: Compute posterior mean ----
        if cond_fn is not None:
            # 有隐式引导: 预测同时做梯度修正 / With guidance: predict + correct
            model_mean, _ = self.apply_cond_fn(
                x, cond, t, index, cond_fn, cfg_scale, uncond
            )
        else:
            # 标准预测: 预测噪声 → 反推 x_0 → 计算后验均值
            # Standard: predict noise → recover x_0 → compute posterior mean
            e_t = self.predict_noise(x, t, cond, cfg_scale, uncond)
            pred_x0 = self._predict_xstart_from_eps(x_t=x, t=index, eps=e_t)
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_x0, x_t=x, t=index
            )

        # ---- 步骤3: 采样 x_{t-1} / Step 3: Sample x_{t-1} ----
        noise = torch.randn_like(x)
        # 最后一步 (index=0) 不加噪声: x_0 就是最终输出
        # Last step (index=0) no noise: x_0 is the final output
        nonzero_mask = (index != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        x_prev = model_mean + nonzero_mask * torch.sqrt(model_variance) * noise
        return x_prev

    # ========================================================================
    # 完整采样循环 / Full Sampling Loop
    # ========================================================================

    @torch.no_grad()
    def sample(
        self,
        steps: int,
        shape: Tuple[int],
        cond: Dict[str, torch.Tensor],
        x_T: Optional[torch.Tensor] = None,
        cfg_scale: float = 1.0,
        cond_fn: Optional[Guidance] = None,
    ) -> torch.Tensor:
        """
        完整去噪采样循环: 纯噪声 → 逐步去噪 → VAE 解码 → 高清图像 /
        Full denoising loop: pure noise → progressive denoising → VAE decode → HR image.

        用法示例 / Example usage:
            >>> sampler = SpacedSampler(model)
            >>> hr = sampler.sample(
            ...     steps=50,
            ...     shape=(1, 4, 60, 60),    # latent 形状 = (B, C, H/8, W/8)
            ...     cond=condition_dict,      # 含 ref 特征 / contains ref features
            ...     cfg_scale=1.0,
            ...     cond_fn=MSEGuidance(...)  # 可选: 用 rwkvSR 输出做结构引导
            ... )

        Args:
            steps (int):        采样步数 (如 50) / Sampling steps (e.g., 50)
            shape (Tuple[int]): latent 形状 (B, C, H, W) / Latent shape
            cond (Dict):        条件字典 / Condition dict
            x_T (Tensor):       可选，指定起始噪声 / Optional, initial noise
            cfg_scale (float):  无分类器引导强度 / CFG scale (>1 = stronger)
            cond_fn (Guidance): 可选，隐式引导 / Optional latent image guidance

        Returns:
            Tensor: 生成的 RGB 高清图像 (B,3,H,W)，值域 [0,1] / Generated HR image
        """
        # ---- 初始化: 构建采样调度 / Init: build sampling schedule ----
        self.make_schedule(num_steps=steps)

        device = next(self.model.parameters()).device
        b = shape[0]
        # 起始纯噪声 / Start from pure Gaussian noise
        img = torch.randn(shape, device=device) if x_T is None else x_T

        # ---- 去噪循环 / Denoising Loop ----
        time_range = np.flip(self.timesteps)  # 从大到小: [999, 979, ..., 19, 0]
        total_steps = len(self.timesteps)
        iterator = tqdm(time_range, desc="Spaced Sampler", total=total_steps)

        for i, step in enumerate(iterator):
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            index = torch.full_like(ts, fill_value=total_steps - i - 1)
            # 单步去噪 / Single denoising step
            img = self.p_sample(
                img, cond, ts,
                index=index, cfg_scale=cfg_scale,
                uncond=None, cond_fn=cond_fn,
            )

        # ---- VAE 解码: latent → RGB 像素 / VAE Decode: latent → RGB pixels ----
        # 从 [-1, 1] 映射到 [0, 1] / Map from [-1, 1] to [0, 1]
        img_pixel = ((self.model.decode_first_stage(img) + 1) / 2).clip(0, 1)
        return img_pixel
