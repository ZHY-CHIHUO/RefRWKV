"""
sd2_ref_adapter.py — 统一 Ref 特征融合适配器 (SD2 UNet)

═══════════════════════════════════════════════════════════════
设计原则: 导入而非迁移 (Import, Don't Migrate)

- 所有核心模块来自 modules.py 和 RefDiffRWKV.py，直接 import 使用
- 本文件仅包含 SD2 特有的组装逻辑和通道适配层
- 不复制任何 encoder、attention、RWKV 算子的实现代码

策略一览:
    ① "lca"     — CAA块 (LocalCrossAttention + MaskAttention) ★CRefDiff 默认
    ② "spade"   — SPADE 归一化调制
    ③ "dual"    — 双路独立编码 + 余弦加权
    ④ "cos_sim" — 编码器内嵌余弦相似度
    ⑤ "cat"     — 拼接后单编码器 (基线)
    ⑥ "rwkv"    — WKV 序列编码 + LCA/MaskAttention 融合 ★RWKV 方法

用法:
    adapter = SD2_RefAdapter(strategy="rwkv")
    f320, f640, f1280 = adapter(LR=lr_image, Ref=ref_image)
    # → 注入 SD2 UNet: h = down[0](h) + f320; ...

依赖:
    - modules.py         → CRefDiff 全部编码器 & 融合模块
    - RefDiffRWKV.py     → RefDiffRWKV (RWKV 序列融合，单尺度输出)
    - RefSRWKV.py        → RUN_CUDA, OmniShift (由 RefDiffRWKV 内部引入)
═══════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

# ═══════════════════════════════════════════════════════════════
#  Part 1: 从 modules.py 导入 CRefDiff 全部模块
# ═══════════════════════════════════════════════════════════════

from modules import (
    cosine_attention_map,
    ConvBlock,
    Resblock,
    Downsample,
    SR_Encoder,
    SR_Ref_Encoder_LCA,
    SR_Ref_Encoder_Spade,
    SR_Ref_Encoder_Cos_Sim,
    ImplicitPromptModule,
)

# ═══════════════════════════════════════════════════════════════
#  Part 2: 从 RefDiffRWKV.py 导入 RWKV 融合模块
# ═══════════════════════════════════════════════════════════════

from RefDiffRWKV import RefDiffRWKV


# ═══════════════════════════════════════════════════════════════
#  Part 3: SD2 特有的组装组件
# ═══════════════════════════════════════════════════════════════

class ResnetBlock_SD2(nn.Module):
    """SD2 风格的残差块 — 用于 CRefDiff/RWKV 系列 Adapter 的后处理链。"""

    def __init__(self, in_c, out_c, down=False, ksize=3, sk=False, use_conv=True):
        super().__init__()
        ps = ksize // 2
        self.in_conv = None if in_c == out_c else nn.Conv2d(in_c, out_c, ksize, 1, ps)
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        self.down = down
        if down:
            self.down_opt = nn.Sequential(
                nn.Conv2d(out_c, out_c, ksize, 1, ps),
                nn.AvgPool2d(2),
            ) if use_conv else nn.AvgPool2d(2)

    def forward(self, x):
        if self.down:
            x = self.down_opt(x)
        if self.in_conv is not None:
            x = self.in_conv(x)
        h = self.block1(x)
        h = self.act(h)
        h = self.block2(h)
        return h + x


class _ConcatEncoderWrapper(nn.Module):
    """
    接口适配器：将单输入 SR_Encoder(in_channel=6) 包装为 (sr, ref) 双参数接口。
    用于 "cat" 策略 — 确保与 CRefDiffAdapterBase 的统一调用约定一致。
    """

    def __init__(self, out_channel: int = 192):
        super().__init__()
        self.encoder = SR_Encoder(out_channel=out_channel, in_channel=6)

    def forward(self, sr: torch.Tensor, ref: torch.Tensor, **kwargs):
        x = torch.cat([sr, ref], dim=1)
        return self.encoder(x)


class CRefDiffAdapterBase(nn.Module):
    """
    CRefDiff / RWKV 系列 Adapter 的统一后处理基类。

    结构: merge_encoder → conv_in → ResnetBody (三级下采样) → [feat0, feat1, feat2]

    merge_encoder 可以来自 modules.py (CNN encoder)，也可以来自 RefDiffRWKV.py (RWKV encoder)，
    只要它接受 (sr, ref) 并返回单尺度特征图 (B, cin, H/8, W/8)。
    """

    def __init__(
        self,
        merge_encoder_class,
        merge_encoder_kwargs: dict,
        channels: tuple = (320, 640, 1280),
        nums_rb: int = 2,
        cin: int = 192,
        ksize: int = 3,
        sk: bool = True,
        use_conv: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.nums_rb = nums_rb
        self.merge_encoder = merge_encoder_class(**merge_encoder_kwargs)
        self.conv_in = nn.Conv2d(cin, channels[0], ksize, 1, ksize // 2)
        self.body = nn.ModuleList()
        for i in range(len(channels) - 1):
            for j in range(nums_rb):
                self.body.append(ResnetBlock_SD2(
                    channels[i] if j > 0 else channels[i],
                    channels[i + 1] if j == 0 else channels[i + 1],
                    down=(j == 0),
                    ksize=ksize,
                    sk=sk,
                    use_conv=use_conv,
                ))

    def forward(self, sr: torch.Tensor, ref: torch.Tensor, **kwargs):
        res = self.merge_encoder(sr, ref, **kwargs)

        if isinstance(res, tuple):
            x = res[0]
        else:
            x = res

        cond_list = []
        x = self.conv_in(x)
        cond_list.append(x)
        for block in self.body:
            x = block(x)
            cond_list.append(x)

        if isinstance(res, tuple) and len(res) > 1:
            return cond_list, res[1:]
        return cond_list


# ═══════════════════════════════════════════════════════════════
#  Part 4: Dual 策略专用实现
#          (双路独立编码 + 余弦加权融合)
# ═══════════════════════════════════════════════════════════════

class _DualBranch(nn.Module):
    """Dual 策略的单分支 — 对 LR 和 Ref 各实例化一个。"""

    def __init__(self, cin: int, channels: tuple, nums_rb: int,
                 ksize: int, sk: bool, use_conv: bool):
        super().__init__()
        self.encoder = SR_Encoder(out_channel=cin, in_channel=3)
        self.conv_in = nn.Conv2d(cin, channels[0], 3, 1, 1)
        self.body = nn.ModuleList()
        for i in range(len(channels) - 1):
            for j in range(nums_rb):
                self.body.append(ResnetBlock_SD2(
                    channels[i] if j > 0 else channels[i],
                    channels[i + 1] if j == 0 else channels[i + 1],
                    down=(j == 0),
                    ksize=ksize,
                    sk=sk,
                    use_conv=use_conv,
                ))

    def forward(self, x):
        x = self.encoder(x)
        feats = [self.conv_in(x)]
        for block in self.body:
            x = block(x)
            feats.append(x)
        return feats


class _DualAdapterImpl(nn.Module):
    """Dual 策略统一封装: 双路独立编码 + 余弦相似度加权融合。"""

    def __init__(self, channels: tuple, nums_rb: int = 2, cin: int = 96,
                 ksize: int = 3, sk: bool = True, use_conv: bool = True):
        super().__init__()
        self.channels = channels
        self.lr_branch = _DualBranch(cin // 2, channels, nums_rb, ksize, sk, use_conv)
        self.ref_branch = _DualBranch(cin // 2, channels, nums_rb, ksize, sk, use_conv)
        self.merger = nn.ModuleList([
            ResnetBlock_SD2(c, c, False, sk=sk) for c in channels
        ])

    def forward(self, sr, ref, sim_lamuda=None,
                return_cos_sim_map=False, **kwargs):
        lr_feats = self.lr_branch(sr)
        ref_feats = self.ref_branch(ref)

        cond_list = []
        cos_maps = []
        for i in range(len(lr_feats)):
            cos_sim = (cosine_attention_map(lr_feats[i], ref_feats[i]) + 1) / 2
            if sim_lamuda is not None:
                cos_sim = cos_sim * sim_lamuda
            cond = cos_sim * ref_feats[i] + (1 - cos_sim) * lr_feats[i]
            cond = self.merger[i](cond)
            cond_list.append(cond)
            if return_cos_sim_map:
                cos_maps.append(cos_sim)

        if return_cos_sim_map:
            return cond_list, cos_maps
        return cond_list


# ═══════════════════════════════════════════════════════════════
#  Part 5: 统一入口 SD2_RefAdapter — 6 种策略一键切换
# ═══════════════════════════════════════════════════════════════

class SD2_RefAdapter(nn.Module):
    """
    ═══════════════════════════════════════════════════════════════
    SD2 统一 Ref 特征融合适配器 — 6 种策略一键切换
    ═══════════════════════════════════════════════════════════════

    策略一览:

    ┌────────┬─────────────────┬───────────────────┬──────────────────┐
    │ ID     │ 策略名          │ 融合机制            │ 范式              │
    ├────────┼─────────────────┼───────────────────┼──────────────────┤
    │  ①     │ "lca"          │ CAA块              │ CNN+Attention    │
    ├────────┼─────────────────┼───────────────────┼──────────────────┤
    │  ②     │ "spade"         │ SPADE调制          │ CNN+SPADE        │
    ├────────┼─────────────────┼───────────────────┼──────────────────┤
    │  ③     │ "dual"          │ 双路+余弦加权       │ CNN双分支        │
    ├────────┼─────────────────┼───────────────────┼──────────────────┤
    │  ④     │ "cos_sim"       │ 内嵌余弦相似度      │ CNN单编码器      │
    ├────────┼─────────────────┼───────────────────┼──────────────────┤
    │  ⑤     │ "cat"           │ 拼接单编码器        │ CNN基线          │
    ├────────┼─────────────────┼───────────────────┼──────────────────┤
    │  ⑥     │ "rwkv"          │ WKV序列编码         │ RWKV序列         │
    │        │                 │ +LCA/MaskAttn      │                  │
    └────────┴─────────────────┴───────────────────┴──────────────────┘

    统一接口:
        adapter = SD2_RefAdapter(strategy="rwkv")
        f320, f640, f1280 = adapter(LR=lr_img, Ref=ref_img)

    Args:
        strategy:    策略名 ("lca" / "spade" / "dual" / "cos_sim" / "cat" / "rwkv")
        channels:    SD2 UNet 目标通道数 (默认 [320, 640, 1280])
        rwkv_cfg:    RWKV 超参数字典 (仅 rwkv 需要)
        **kwargs:    传递给具体 Adapter 的额外参数
    """

    STRATEGIES = ["lca", "spade", "dual", "cos_sim", "cat", "rwkv"]

    # 编码器映射表 — 所有 encoder 类均来自 modules.py 或 RefDiffRWKV.py
    _ENCODER_MAP = {
        "lca":     (SR_Ref_Encoder_LCA,     {"out_channel": 192, "in_sr_channel": 3, "in_ref_channel": 3}),
        "spade":   (SR_Ref_Encoder_Spade,    {"out_channel": 192}),
        "dual":    (SR_Encoder,              {"out_channel": 192, "in_channel": 3}),  # Dual 自己处理双路
        "cos_sim": (SR_Ref_Encoder_Cos_Sim,  {"out_channel": 192}),
        "cat":     (_ConcatEncoderWrapper,   {"out_channel": 192}),
        "rwkv":    (RefDiffRWKV,             {"out_channel": 192}),
    }

    def __init__(
        self,
        strategy: str = "rwkv",
        channels: tuple = (320, 640, 1280),
        sd2_dims: Optional[tuple] = None,
        rwkv_cfg: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__()

        if strategy not in self.STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Must be one of {self.STRATEGIES}"
            )

        self.strategy = strategy
        self.channels = channels
        self.sd2_dims = sd2_dims or channels

        if strategy == "dual":
            # ═══ Dual 路径: 特殊双路结构 ═══
            self.core = _DualAdapterImpl(
                channels=channels,
                nums_rb=kwargs.get("nums_rb", 2),
                cin=kwargs.get("cin", 192),
                ksize=kwargs.get("ksize", 3),
                sk=kwargs.get("sk", True),
                use_conv=kwargs.get("use_conv", True),
            )
            self.adapter_type = "crefdiff"

        elif strategy == "rwkv":
            # ═══ RWKV 路径: RefDiffRWKV 作为 encoder，统一走后处理链 ═══
            cfg = rwkv_cfg or {}
            self.core = CRefDiffAdapterBase(
                merge_encoder_class=RefDiffRWKV,
                merge_encoder_kwargs={
                    "out_channel": 192,
                    "in_channels": cfg.get("in_channels", 3),
                    "patch_size": cfg.get("patch_size", 4),
                    "embed_dim": cfg.get("embed_dim", 384),
                    "upsample_mode": cfg.get("upsample_mode", "bilinear"),
                },
                channels=channels,
                nums_rb=kwargs.get("nums_rb", 2),
                cin=kwargs.get("cin", 192),
                ksize=kwargs.get("ksize", 3),
                sk=kwargs.get("sk", True),
                use_conv=kwargs.get("use_conv", True),
            )
            self.adapter_type = "rwkv"

        else:
            # ═══ 其他 CRefDiff 路径: 统一基类 ═══
            encoder_cls, encoder_kw = self._ENCODER_MAP[strategy]
            self.core = CRefDiffAdapterBase(
                merge_encoder_class=encoder_cls,
                merge_encoder_kwargs=encoder_kw,
                channels=channels,
                nums_rb=kwargs.get("nums_rb", 2),
                cin=kwargs.get("cin", 192),
                ksize=kwargs.get("ksize", 3),
                sk=kwargs.get("sk", True),
                use_conv=kwargs.get("use_conv", True),
            )
            self.adapter_type = "crefdiff"

    def forward(
        self,
        LR: torch.Tensor,
        Ref: torch.Tensor,
        **kwargs,
    ) -> Tuple[torch.Tensor, ...]:
        """
        统一前向传播。

        Args:
            LR:  低分辨率图像 (B, 3, h, w)
            Ref: 参考图像 (B, 3, H, W)
            **kwargs:
                - sim_lamuda:           相似度缩放系数
                - return_cos_sim_map:   是否返回余弦相似度图
                - return_learned_sim_map: 是否返回学习的 mask 图

        Returns:
            (feat_ch1, feat_ch2, feat_ch3) 三尺度控制信号，
            通道数由 channels 决定。
        """
        if self.strategy == "dual":
            return self.core(LR, Ref, **kwargs)
        else:
            return self.core(sr=LR, ref=Ref, **kwargs)

    def get_info(self) -> dict:
        """返回当前配置摘要。"""
        info = {
            "strategy": self.strategy,
            "type": self.adapter_type,
            "channels": self.channels,
            "sd2_dims": self.sd2_dims,
            "params": sum(p.numel() for p in self.parameters()),
        }
        if self.strategy == "rwkv":
            info["encoder_params"] = sum(
                p.numel() for p in self.core.merge_encoder.parameters()
            )
        return info

    def __repr__(self):
        i = self.get_info()
        return (
            f"SD2_RefAdapter(strategy={self.strategy!r}, "
            f"type={i['type']}, params={i['params']:,})"
        )


# ═══════════════════════════════════════════════════════════════
#  Part 6: 工具函数 & 测试入口
# ═══════════════════════════════════════════════════════════════

def build_all_adapters(
    channels=(320, 640, 1280),
    sd2_dims=None,
    rwkv_cfg=None,
    **common_kw,
) -> dict:
    """一键构建全部 6 种策略的 Adapter。"""
    models = {}
    for s in SD2_RefAdapter.STRATEGIES:
        try:
            kw = dict(**common_kw)
            if s == "rwkv":
                kw["rwkv_cfg"] = rwkv_cfg or {}
            models[s] = SD2_RefAdapter(strategy=s, channels=channels, **kw)
        except Exception as e:
            print(f"[WARN] Failed to build '{s}': {e}")
            models[s] = None
    return models


def print_comparison(models: dict):
    """打印各策略参数量对比表。"""
    print("=" * 80)
    print(f"{'Strategy':<12} {'Type':<12} {'Params':>12} {'Status'}")
    print("-" * 80)
    for name in SD2_RefAdapter.STRATEGIES:
        m = models.get(name)
        if m is None:
            print(f"{name:<12} {'N/A':<12} {'N/A':>12} ❌ Build failed")
        else:
            i = m.get_info()
            print(f"{name:<12} {i['type']:<12} {i['params']:>12,} ✅ Ready")
    print("=" * 80)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n" + "=" * 80)
    print("  SD2_RefAdapter — 6合1 统一 Ref 特征融合框架 测试")
    print("=" * 80)

    models = build_all_adapters(
        channels=(320, 640, 1280),
        rwkv_cfg={"patch_size": 4, "embed_dim": 384},
    )
    print_comparison(models)

    # 测试 RWKV 策略
    print("\n--- Testing RWKV strategy ---")
    rwkv_model = models.get("rwkv")
    if rwkv_model is not None:
        rwkv_model = rwkv_model.to(device)
        B = 2
        LR = torch.randn(B, 3, 48, 48).to(device)
        Ref = torch.randn(B, 3, 480, 480).to(device)
        with torch.no_grad():
            f1, f2, f3 = rwkv_model(LR=LR, Ref=Ref)
        print(f"  Input:  LR={list(LR.shape)}, Ref={list(Ref.shape)}")
        print(f"  Output: f1={list(f1.shape)}, f2={list(f2.shape)}, f3={list(f3.shape)}")
        print(f"  Params: {sum(p.numel() for p in rwkv_model.parameters()):,}")
        print("  ✅ RWKV test passed!")
    else:
        print("  ❌ RWKV build failed!")

    # 测试 CRefDiff 策略
    print("\n--- Testing CRefDiff strategies ---")
    for name in ["lca", "spade", "dual", "cos_sim", "cat"]:
        m = models.get(name)
        if m is not None:
            print(f"  ✅ {name}: {m}")
        else:
            print(f"  ⚠️  {name}: skipped (module import failed)")

    print("\n" + "=" * 80)
    print("  Usage:")
    print("  >>> adapter = SD2_RefAdapter(strategy='rwkv')")
    print("  >>> f320, f640, f1280 = adapter(LR=lr_img, Ref=ref_img)")
    print("  >>> # Switch:")
    print("  >>> adapter = SD2_RefAdapter(strategy='lca')")
    print("=" * 80)
