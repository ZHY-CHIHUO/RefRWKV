import copy

import torch.nn as nn

from .vss_blocks import dual_vss_block


class MambaSRmodel(nn.Module):
    def __init__(self, channel_first=False, norm_layer="LN", ssm_act_layer="silu", mlp_act_layer="gelu", **kwargs):
        super().__init__()
        in_chans = int(kwargs.get("in_chans", 4))
        hidden_dims = [12, 64, 12]
        self.conv_first = nn.Conv2d(in_chans, hidden_dims[0], kernel_size=3, stride=2, padding=1)
        self.sr_block = nn.ModuleList(
            [
                dual_vss_block(
                    hidden_dims[1],
                    hidden_dims[2],
                    channel_first=channel_first,
                    norm_layer=norm_layer,
                    ssm_act_layer=ssm_act_layer,
                    mlp_act_layer=mlp_act_layer,
                    kwargs=kwargs,
                    in_channels=hidden_dims[0],
                    output_dim=hidden_dims[0],
                )
                for _ in range(2)
            ]
        )
        self.conv_final = nn.ConvTranspose2d(hidden_dims[0], in_chans, kernel_size=4, stride=2, padding=1)
        self.sr_block_list = nn.ModuleList([copy.deepcopy(self.sr_block) for _ in range(4)])

    def forward(self, x):
        input_residual = x
        x = self.conv_first(x)
        for i in range(4):
            residual = x
            for block in self.sr_block_list[i]:
                x = block(x)
            x = x + residual
        x = self.conv_final(x)
        return x + input_residual
