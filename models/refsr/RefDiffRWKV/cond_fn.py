"""
cond_fn.py — Diffusion Sampling Gradient Guidance (扩散采样梯度引导)

在 SpacedSampler 采样过程中，利用目标图像（rwkvSR 输出）对预测 x0 施加 MSE 梯度，
迫使最终 SR 结果在像素/潜在空间上更接近第一阶段的初步超分图像。

References:
    Generative Diffusion Prior (GDP): https://github.com/Fayeben/GenerativeDiffusionPrior
"""

from typing import overload, Optional
import torch
from torch.nn import functional as F


class Guidance:
    """
    Base class for classifier-free-style latent image guidance (潜在图像引导基类).

    核心思想：
    ┌─────────────────────────────────────────────────────────────┐
    │ 在采样 t ∈ (t_start, t_stop) 区间内：
    │   1. 拿到当前 timestep 预测的 pred_x0（去噪后的干净图像估计）
    │   2. 与预先加载的目标 target_x0 计算损失（如 MSE）
    │   3. 对 pred_x0 求梯度，作为修正方向注入采样过程
    │
    │ 最终效果：生成的 SR 图像在指定区间被"拉向"目标图像
    └─────────────────────────────────────────────────────────────┘
    """

    def __init__(
        self,
        scale: float,  # 梯度缩放系数（论文中的 s），越大则输出越接近第一阶段结果
        t_start: int,  # 引导起始 timestep（靠近纯噪声端，通常较大，如 1000）
        t_stop: int,  # 引导停止 timestep（靠近干净图像端，通常较小，如 200）
        space: str,  # 损失计算空间："rgb" 像素空间 / "latent" 潜在空间
        repeat: int,  # 每个 timestep 重复梯度下降次数
    ) -> "Guidance":
        """
        Initialize latent image guidance.

        Args:
            scale:
                Gradient scale (denoted as `s` in the GDP paper).
                Larger scale → output more strictly matches the target (rwkvSR output).
                梯度缩放系数。值越大，最终输出越接近目标图像。
            t_start:
                Timestep to start guidance. Sampling goes from T→0,
                so t_start should be > t_stop.
                引导开始的 timestep。采样从 T 到 0，因此 t_start > t_stop。
            t_stop:
                Timestep to stop guidance. Early stopping prevents over-sharpening.
                引导停止的 timestep。提前停止可避免过度锐化/伪影。
            space:
                Data space for computing the loss function.
                - "rgb":   compute MSE in RGB pixel space after VAE decode
                - "latent": compute MSE directly in latent space
                损失计算空间，"rgb" 在像素空间，"latent" 在潜在空间。
            repeat:
                Number of gradient descent repeats per timestep.
                More repeats = stronger guidance per step.
                每个 timestep 内重复梯度修正的次数。
        """
        self.scale = scale
        self.t_start = t_start
        self.t_stop = t_stop
        self.target = None  # 目标图像，由外部调用 load_target() 设置
        self.space = space
        self.repeat = repeat

    def load_target(self, target: torch.Tensor) -> None:
        """
        Pre-load the target image before sampling starts.
        在采样前预加载目标图像（通常是 rwkvSR 的输出）。

        Args:
            target: Target image tensor, shape (B, C, H, W), value range [-1, 1].
                    目标图像张量，值域 [-1, 1]。
        """
        self.target = target

    def __call__(
        self,
        target_x0: torch.Tensor,  # 目标干净图像（来自 rwkvSR）
        pred_x0: torch.Tensor,  # 当前 timestep 预测的干净图像
        t: int,  # 当前 timestep
    ) -> Optional[torch.Tensor]:
        """
        Called by SpacedSampler at each denoising step.
        由 SpacedSampler 在每个去噪步骤中调用。

        Logic (逻辑):
            1. 仅在 t ∈ (t_stop, t_start) 区间内生效
            2. 先 detach 切断计算图（防止梯度污染采样主流程）
            3. 调用 _forward 计算梯度
            4. 返回 scale * gradient 作为修正量

        Returns:
            Gradient correction tensor if in range, else None.
            区间内返回梯度修正张量，区间外返回 None（跳过引导）。
        """
        if self.t_stop < t < self.t_start:
            # In guidance range — apply gradient correction (在引导区间内，施加梯度修正)
            # Detach both tensors to isolate guidance gradient from the main sampling graph
            # 两个张量都 detach，确保引导梯度不会污染采样主流程的计算图
            pred_x0 = pred_x0.detach().clone()
            target_x0 = target_x0.detach().clone()
            return self.scale * self._forward(target_x0, pred_x0)
        else:
            # Outside guidance range — skip (超出引导区间，不做修正)
            return None

    @overload
    def _forward(self, target_x0: torch.Tensor, pred_x0: torch.Tensor) -> torch.Tensor:
        """
        Subclass must implement: compute loss → return gradient w.r.t. pred_x0.
        子类必须实现：计算损失 → 返回 pred_x0 的梯度。
        """
        ...


class MSEGuidance(Guidance):
    """
    MSE-based latent image guidance (基于 MSE 损失的潜在图像引导).

    Loss:
        L = || pred_x0 - target_x0 ||^2        (per-sample, then sum over batch)

    Gradient:
        ∂L/∂(pred_x0) = 2 * (pred_x0 - target_x0)

    We return the NEGATIVE gradient (-∂L/∂pred_x0) because the sampler
    uses gradient ASCENT to maximize similarity (minimize MSE).
    返回负梯度（-∂L/∂pred_x0），因为采样器做梯度上升来最大化相似度。

    Usage in your pipeline (在你的管线中的用法):
        guidance = MSEGuidance(scale=100.0, t_start=800, t_stop=200,
                               space="rgb", repeat=1)
        guidance.load_target(rwkvSR_output)   # 目标 = 第一阶段超分结果
        # SpacedSampler 会自动在每个步骤调用 guidance(target_x0, pred_x0, t)
    """

    def __init__(
        self, scale: float, t_start: int, t_stop: int, space: str, repeat: int
    ) -> "MSEGuidance":
        super().__init__(scale, t_start, t_stop, space, repeat)

    @torch.enable_grad()
    def _forward(
        self,
        target_x0: torch.Tensor,  # (B, C, H, W), [-1, 1], rgb or latent
        pred_x0: torch.Tensor,  # (B, C, H, W), [-1, 1], rgb or latent
    ) -> torch.Tensor:
        """
        Compute MSE loss between prediction and target, then return the
        NEGATIVE gradient w.r.t. pred_x0 (for gradient ASCENT).
        计算预测与目标之间的 MSE 损失，返回 pred_x0 的负梯度（用于梯度上升）。

        Workflow (工作流):
            1. Enable grad on pred_x0               ← 开启梯度追踪
            2. Compute per-sample MSE, sum over batch ← 逐样本 MSE 求和
            3. Compute gradient via autograd        ← 自动求导
            4. Return -grad (ascent direction)      ← 返回负梯度（上升方向）

        Args:
            target_x0: Target clean image (first-stage SR output).
                       目标干净图像（第一阶段 rwkvSR 输出）。
            pred_x0:   Predicted clean image at current timestep.
                       当前 timestep 预测的干净图像。

        Returns:
            Gradient correction tensor, same shape as pred_x0.
            梯度修正张量，形状与 pred_x0 相同。
        """
        # inputs: [-1, 1], nchw, rgb (or latent)
        # 输入值域 [-1, 1]，形状 (N, C, H, W)，RGB 或潜在空间

        pred_x0.requires_grad_(True)

        # Per-sample MSE, then sum over batch
        # 逐样本计算 MSE（对 C, H, W 维度取均值），再对 batch 求和
        # shape: (B, C, H, W) → mean over (1,2,3) → (B,) → sum → scalar
        loss = (pred_x0 - target_x0).pow(2).mean((1, 2, 3)).sum()

        print(f"[MSEGuidance] t={self.t_start}~{self.t_stop}, loss = {loss.item():.6f}")

        # Return NEGATIVE gradient:
        # The sampler performs x ← x + guidance_grad (gradient ASCENT)
        # We want to MINIMIZE MSE → gradient ASCENT on -MSE = DESCENT on MSE
        # ∂(-MSE)/∂pred_x0 = -∂MSE/∂pred_x0
        # 返回负梯度：采样器做梯度上升（x ← x + grad），
        # 我们想最小化 MSE → 相当于对 -MSE 做梯度上升
        return -torch.autograd.grad(loss, pred_x0)[0]
