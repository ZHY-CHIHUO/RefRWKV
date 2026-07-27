"""
sd2_ref_adapter.py — 统一 Ref 特征融合适配器 (SD2 UNet)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List

from .modules import (
    cosine_attention_map,
    ConvBlock,
    Resblock,
    Downsample,
    SR_Encoder,
    SR_Ref_Encoder_LCA,
    ImplicitPromptModule,
)

from .RefDiffRWKV import RefDiffRWKV


class ResnetBlock_SD2(nn.Module):
    def __init__(self, in_c, out_c, down=False, ksize=3, sk=False, use_conv=True):
        super().__init__()
        ps = ksize // 2
        self.in_conv = None if in_c == out_c else nn.Conv2d(in_c, out_c, ksize, 1, ps)
        self.block1 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.act = nn.ReLU()
        self.block2 = nn.Conv2d(out_c, out_c, ksize, 1, ps)
        self.down = down

        if down:
            if use_conv:
                self.down_opt = nn.Sequential(
                    nn.Conv2d(in_c, in_c, ksize, 1, ps),
                    nn.AvgPool2d(2),
                )
            else:
                self.down_opt = nn.AvgPool2d(2)

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
    def __init__(self, out_channel: int = 192):
        super().__init__()
        self.encoder = SR_Encoder(out_channel=out_channel, in_channel=6)

    def forward(self, sr: torch.Tensor, ref: torch.Tensor, **kwargs):
        x = torch.cat([sr, ref], dim=1)
        return self.encoder(x)


class DiffAdapterBase(nn.Module):
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
                self.body.append(
                    ResnetBlock_SD2(
                        channels[i] if j == 0 else channels[i + 1],  # 修复点 1
                        channels[i + 1],
                        down=(j == 0),
                        ksize=ksize,
                        sk=sk,
                        use_conv=use_conv,
                    )
                )

    def forward(self, sr: torch.Tensor, ref: torch.Tensor, **kwargs):
        res = self.merge_encoder(sr, ref, **kwargs)

        if isinstance(res, tuple):
            x = res[0]
        else:
            x = res

        cond_list = []
        x = self.conv_in(x)
        cond_list.append(x)

        block_idx = 0
        for i in range(len(self.channels) - 1):
            for j in range(self.nums_rb):
                x = self.body[block_idx](x)
                block_idx += 1
            cond_list.append(x)  # 每个 stage 结束后再收集

        if isinstance(res, tuple) and len(res) > 1:
            extra = res[1]  # cos_maps 本体（list），不是 (list,) 元组
            return cond_list, extra
        return cond_list


class _DualBranch(nn.Module):
    def __init__(
        self,
        cin: int,
        channels: tuple,
        nums_rb: int,
        ksize: int,
        sk: bool,
        use_conv: bool,
    ):
        super().__init__()
        self.channels = channels
        self.encoder = SR_Encoder(out_channel=cin, in_channel=3)
        self.conv_in = nn.Conv2d(cin, channels[0], 3, 1, 1)
        self.body = nn.ModuleList()

        for i in range(len(channels) - 1):
            for j in range(nums_rb):
                in_c = channels[i] if j == 0 else channels[i + 1]
                out_c = channels[i + 1]
                self.body.append(
                    ResnetBlock_SD2(
                        in_c,
                        out_c,
                        down=(j == 0),
                        ksize=ksize,
                        sk=sk,
                        use_conv=use_conv,
                    )
                )

    def forward(self, x):
        x = self.encoder(x)
        feats = [self.conv_in(x)]

        block_idx = 0
        for i in range(len(self.channels) - 1):
            for j in range(self.nums_rb):
                x = self.body[block_idx](x)
                block_idx += 1
            feats.append(x)  # 每个 stage 结束后再收集

        return feats


class _DualAdapterImpl(nn.Module):
    def __init__(
        self,
        channels: tuple,
        nums_rb: int = 2,
        cin: int = 96,
        ksize: int = 3,
        sk: bool = True,
        use_conv: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.lr_branch = _DualBranch(cin // 2, channels, nums_rb, ksize, sk, use_conv)
        self.ref_branch = _DualBranch(cin // 2, channels, nums_rb, ksize, sk, use_conv)
        self.merger = nn.ModuleList(
            [ResnetBlock_SD2(c, c, False, sk=sk) for c in channels]
        )

    def forward(self, sr, ref, sim_lamuda=None, return_cos_sim_map=False, **kwargs):
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


class SD2_RefAdapter(nn.Module):
    STRATEGIES = ["lca", "dual", "cat", "rwkv"]

    _ENCODER_MAP = {
        "lca": (
            SR_Ref_Encoder_LCA,
            {"out_channel": 192, "in_sr_channel": 3, "in_ref_channel": 3},
        ),
        "dual": (SR_Encoder, {"out_channel": 192, "in_channel": 3}),
        "cat": (_ConcatEncoderWrapper, {"out_channel": 192}),
        "rwkv": (RefDiffRWKV, {"out_channel": 192}),
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
            cfg = rwkv_cfg or {}
            self.core = DiffAdapterBase(
                merge_encoder_class=RefDiffRWKV,
                merge_encoder_kwargs={
                    "out_channel": 192,
                    "in_channels": cfg.get("in_channels", 3),
                    "patch_size": cfg.get("patch_size", 4),
                    "embed_dim": cfg.get("embed_dim", 384),
                    "upsample_mode": cfg.get("upsample_mode", "bilinear"),
                    "ref_size": cfg.get("ref_size", 480),
                    "use_self_sim_transfer": cfg.get("use_self_sim_transfer", False),
                    "self_sim_topk": cfg.get("self_sim_topk", 8),
                    "self_sim_init_alpha": cfg.get("self_sim_init_alpha", 0.3),
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
            encoder_cls, encoder_kw = self._ENCODER_MAP[strategy]
            self.core = DiffAdapterBase(
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
        self, LR: torch.Tensor, Ref: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        if self.strategy == "dual":
            return self.core(LR, Ref, **kwargs)
        else:
            return self.core(sr=LR, ref=Ref, **kwargs)

    def get_info(self) -> dict:
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


def build_all_adapters(
    channels=(320, 640, 1280),
    sd2_dims=None,
    rwkv_cfg=None,
    **common_kw,
) -> dict:
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
        rwkv_cfg={"patch_size": 4, "embed_dim": 192},
    )
    print_comparison(models)

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
        print(
            f"  Output: f1={list(f1.shape)}, f2={list(f2.shape)}, f3={list(f3.shape)}"
        )
        print(f"  Params: {sum(p.numel() for p in rwkv_model.parameters()):,}")
        print("  ✅ RWKV test passed!")
    else:
        print("  ❌ RWKV build failed!")

    print("\n--- Testing CRefDiff strategies ---")
    for name in ["lca", "dual", "cat"]:
        m = models.get(name)
        if m is not None:
            print(f"  ✅ {name}: {m}")
        else:
            print(f"  ⚠️  {name}: skipped (module import failed)")

    print("\n" + "=" * 80)
