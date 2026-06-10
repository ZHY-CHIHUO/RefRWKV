# Copyright (c) Shanghai AI Lab. All rights reserved.
import math, os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F 
from einops import rearrange
from torch.utils.cpp_extension import load
wkv_cuda = load(name="bi_wkv", sources=["./cuda/bi_wkv.cpp", "./cuda/bi_wkv_kernel.cu"],
                verbose=True, extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', '-gencode arch=compute_86,code=sm_86'])


class WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, u, k, v):
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        ctx.save_for_backward(w, u, k, v)
        w = w.float().contiguous()
        u = u.float().contiguous()
        k = k.float().contiguous()
        v = v.float().contiguous()
        y = wkv_cuda.bi_wkv_forward(w, u, k, v)
        if half_mode:
            y = y.half()
        elif bf_mode:
            y = y.bfloat16()
        return y

    @staticmethod
    def backward(ctx, gy):
        w, u, k, v = ctx.saved_tensors
        half_mode = (w.dtype == torch.half)
        bf_mode = (w.dtype == torch.bfloat16)
        gw, gu, gk, gv = wkv_cuda.bi_wkv_backward(w.float().contiguous(),
                          u.float().contiguous(),
                          k.float().contiguous(),
                          v.float().contiguous(),
                          gy.float().contiguous())
        if half_mode:
            return (gw.half(), gu.half(), gk.half(), gv.half())
        elif bf_mode:
            return (gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())
        else:
            return (gw, gu, gk, gv)


def RUN_CUDA(w, u, k, v):
    return WKV.apply(w.cuda(), u.cuda(), k.cuda(), v.cuda())




class OmniShift(nn.Module):
    def __init__(self, dim):
        super(OmniShift, self).__init__()
        # Define the layers for training
        self.conv1x1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, groups=dim, bias=False)
        self.conv3x3 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim, bias=False)
        self.conv5x5 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, padding=2, groups=dim, bias=False) 
        self.alpha = nn.Parameter(torch.randn(4), requires_grad=True) 
        

        # Define the layers for testing
        self.conv5x5_reparam = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, padding=2, groups=dim, bias = False) 
        self.repram_flag = True

    def forward_train(self, x):
        out1x1 = self.conv1x1(x)
        out3x3 = self.conv3x3(x)
        out5x5 = self.conv5x5(x) 
        # import pdb 
        # pdb.set_trace() 
        
        
        out = self.alpha[0]*x + self.alpha[1]*out1x1 + self.alpha[2]*out3x3 + self.alpha[3]*out5x5
        return out

    def reparam_5x5(self):
        # Combine the parameters of conv1x1, conv3x3, and conv5x5 to form a single 5x5 depth-wise convolution 
        
        padded_weight_1x1 = F.pad(self.conv1x1.weight, (2, 2, 2, 2)) 
        padded_weight_3x3 = F.pad(self.conv3x3.weight, (1, 1, 1, 1)) 
        
        identity_weight = F.pad(torch.ones_like(self.conv1x1.weight), (2, 2, 2, 2)) 
        
        combined_weight = self.alpha[0]*identity_weight + self.alpha[1]*padded_weight_1x1 + self.alpha[2]*padded_weight_3x3 + self.alpha[3]*self.conv5x5.weight 
        
        device = self.conv5x5_reparam.weight.device 

        combined_weight = combined_weight.to(device)

        self.conv5x5_reparam.weight = nn.Parameter(combined_weight)


    def forward(self, x): 
        
        if self.training: 
            self.repram_flag = True
            out = self.forward_train(x) 
        elif self.training == False and self.repram_flag == True:
            self.reparam_5x5() 
            self.repram_flag = False 
            out = self.conv5x5_reparam(x)
        elif self.training == False and self.repram_flag == False:
            out = self.conv5x5_reparam(x)
        
        return out 



class VRWKV_SpatialMix(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.n_embd = n_embd
        self.device = None
        attn_sz = n_embd
        
        self.recurrence = 2 
        
        self.omni_shift = OmniShift(dim=n_embd)


        self.key = nn.Linear(n_embd, attn_sz, bias=False)
        self.value = nn.Linear(n_embd, attn_sz, bias=False)
        self.receptance = nn.Linear(n_embd, attn_sz, bias=False)
        self.output = nn.Linear(attn_sz, n_embd, bias=False) 


        with torch.no_grad():
            self.spatial_decay = nn.Parameter(torch.randn((self.recurrence, self.n_embd))) 
            self.spatial_first = nn.Parameter(torch.randn((self.recurrence, self.n_embd))) 



    def jit_func(self, x, resolution):
        # Mix x with the previous timestep to produce xk, xv, xr

        
        h, w = resolution

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.omni_shift(x)
        x = rearrange(x, 'b c h w -> b (h w) c')    


        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)
        sr = torch.sigmoid(r)

        return sr, k, v


        
        

    def forward(self, x, resolution):
        B, T, C = x.size()
        self.device = x.device

        sr, k, v = self.jit_func(x, resolution) 
        
        for j in range(self.recurrence): 
            if j%2==0:
                v = RUN_CUDA(self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v) 
            else:
                h, w = resolution 
                k = rearrange(k, 'b (h w) c -> b (w h) c', h=h, w=w) 
                v = rearrange(v, 'b (h w) c -> b (w h) c', h=h, w=w) 
                v = RUN_CUDA(self.spatial_decay[j] / T, self.spatial_first[j] / T, k, v) 
                k = rearrange(k, 'b (w h) c -> b (h w) c', h=h, w=w) 
                v = rearrange(v, 'b (w h) c -> b (h w) c', h=h, w=w) 
                

        x = v
        x = sr * x
        x = self.output(x)
        return x



class VRWKV_ChannelMix(nn.Module):
    def __init__(self, n_embd, hidden_rate=4):
        super().__init__()
        self.n_embd = n_embd
        hidden_sz = int(hidden_rate * n_embd)
        self.key = nn.Linear(n_embd, hidden_sz, bias=False) 
        
        self.omni_shift = OmniShift(dim=n_embd)
        self.receptance = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(hidden_sz, n_embd, bias=False)



    def forward(self, x, resolution):

        h, w = resolution

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.omni_shift(x)
        x = rearrange(x, 'b c h w -> b (h w) c')    


        k = self.key(x)
        k = torch.square(torch.relu(k))
        kv = self.value(k)
        x = torch.sigmoid(self.receptance(x)) * kv 

        return x



class Block(nn.Module):
    def __init__(self, n_embd, hidden_rate=4):
        super().__init__()

         

        
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd) 


        self.att = VRWKV_SpatialMix(n_embd)

        self.ffn = VRWKV_ChannelMix(n_embd, hidden_rate)



        self.gamma1 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)
        self.gamma2 = nn.Parameter(torch.ones((n_embd)), requires_grad=True)


    def forward(self, x): 
        b, c, h, w = x.shape
        
        resolution = (h, w)

        x = rearrange(x, 'b c h w -> b (h w) c')
        x = x + self.gamma1 * self.att(self.ln1(x), resolution) 
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
    
        x = rearrange(x, 'b c h w -> b (h w) c')    
        x = x + self.gamma2 * self.ffn(self.ln2(x), resolution) 
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        return x



##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class EnRWKV(nn.Module):
    """
    基于视觉RWKV的U-Net图像恢复模型。

    整体结构：
        - 浅层特征提取（patch_embed）
        - 四级编码器，每级由多个 VRWKV Block 组成，逐级下采样
        - 底层 latent 处理
        - 四级解码器，通过跳跃连接与编码器特征融合，逐级上采样
        - 细化模块（refinement）
        - 输出层 + 全局残差连接
    """

    def __init__(self,
                 inp_channels=1,            # 输入图像通道数（默认灰度图1）
                 out_channels=1,            # 输出图像通道数
                 dim=48,                    # 第一层特征通道数（基础宽度）
                 num_blocks=[4, 6, 6, 8],   # 各编码/解码级的 Block 数量，依次为 level1 ~ level4
                 num_refinement_blocks=4,   # 解码器末尾额外添加的细化 Block 数量
                 loss_fun=nn.L1Loss()       # 训练损失函数，默认 L1
                 ):
        super(EnRWKV, self).__init__()

        # ===== 浅层特征提取 =====
        # 3x3卷积，将输入图像映射到 dim 维度，保持分辨率
        self.patch_embed = nn.Conv2d(inp_channels, dim, kernel_size=3, stride=1, padding=1, bias=True)

        self.loss_fun = loss_fun

        # ===== 编码器（Encoder）=====
        # Level 1: 输入 dim 通道，输出 dim 通道，分辨率不变
        self.encoder_level1 = nn.Sequential(
            *[Block(n_embd=dim) for _ in range(num_blocks[0])]
        )

        # 下采样 Level1 → Level2 : 分辨率减半，通道数变为 dim*2
        self.down1_2 = Downsample(dim)   # 输入dim，输出dim*2
        self.encoder_level2 = nn.Sequential(
            *[Block(n_embd=int(dim * 2 ** 1)) for _ in range(num_blocks[1])]
        )

        # 下采样 Level2 → Level3 : 通道数变为 dim*4
        self.down2_3 = Downsample(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(
            *[Block(n_embd=int(dim * 2 ** 2)) for _ in range(num_blocks[2])]
        )

        # 下采样 Level3 → Level4 (latent): 通道数变为 dim*8
        self.down3_4 = Downsample(int(dim * 2 ** 2))
        self.latent = nn.Sequential(
            *[Block(n_embd=int(dim * 2 ** 3)) for _ in range(num_blocks[3])]
        )

        # ===== 解码器（Decoder）=====
        # 上采样 Level4 → Level3 : 分辨率加倍，通道数变为 dim*4
        self.up4_3 = Upsample(int(dim * 2 ** 3))          # 输入dim*8，输出dim*4
        # 将拼接后的 dim*8 通道压缩为 dim*4
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=True)
        self.decoder_level3 = nn.Sequential(
            *[Block(n_embd=int(dim * 2 ** 2)) for _ in range(num_blocks[2])]
        )

        # 上采样 Level3 → Level2 : 输出 dim*2
        self.up3_2 = Upsample(int(dim * 2 ** 2))          # 输入dim*4，输出dim*2
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=True)
        self.decoder_level2 = nn.Sequential(
            *[Block(n_embd=int(dim * 2 ** 1)) for _ in range(num_blocks[1])]
        )

        # 上采样 Level2 → Level1 : 输出 dim（与 encoder_level1 通道数相同）
        self.up2_1 = Upsample(int(dim * 2 ** 1))          # 输入dim*2，输出dim
        # 拼接后通道数为 dim*2，直接送入 Block，不再用 1x1 降维
        self.decoder_level1 = nn.Sequential(
            *[Block(n_embd=int(dim * 2 ** 1)) for _ in range(num_blocks[0])]
        )

        # ===== 细化模块 =====
        self.refinement = nn.Sequential(
            *[Block(n_embd=int(dim * 2 ** 1)) for _ in range(num_refinement_blocks)]
        )

        # ===== 输出层 =====
        # 将 dim*2 特征映射到输出通道，并加上输入形成全局残差
        self.output = nn.Conv2d(int(dim * 2 ** 1), out_channels, kernel_size=3, stride=1, padding=1, bias=True)


    def forward(self, inp_img, label=None):
        """
        前向传播。

        参数:
            inp_img: 输入图像，shape (B, C, H, W)
            label:   Ground truth 图像，若不为 None 则直接返回损失值，否则返回增强结果

        返回:
            若 label 不为 None，返回损失标量；
            否则返回恢复图像，shape (B, out_channels, H, W)
        """

        # ---------- 编码 ----------
        inp_enc_level1 = self.patch_embed(inp_img)                 # (B, dim, H, W)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)       # (B, dim, H, W)

        inp_enc_level2 = self.down1_2(out_enc_level1)              # (B, dim*2, H/2, W/2)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)       # (B, dim*2, H/2, W/2)

        inp_enc_level3 = self.down2_3(out_enc_level2)              # (B, dim*4, H/4, W/4)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)       # (B, dim*4, H/4, W/4)

        inp_enc_level4 = self.down3_4(out_enc_level3)              # (B, dim*8, H/8, W/8)
        latent = self.latent(inp_enc_level4)                       # (B, dim*8, H/8, W/8)

        # ---------- 解码（带跳跃连接）----------
        # Level4 -> Level3
        inp_dec_level3 = self.up4_3(latent)                        # (B, dim*4, H/4, W/4)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], dim=1)  # (B, dim*8, H/4, W/4)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)   # (B, dim*4, H/4, W/4)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)       # (B, dim*4, H/4, W/4)

        # Level3 -> Level2
        inp_dec_level2 = self.up3_2(out_dec_level3)                # (B, dim*2, H/2, W/2)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=1)  # (B, dim*4, H/2, W/2)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)   # (B, dim*2, H/2, W/2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)       # (B, dim*2, H/2, W/2)

        # Level2 -> Level1
        inp_dec_level1 = self.up2_1(out_dec_level2)                # (B, dim, H, W)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=1)  # (B, dim*2, H, W)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)       # (B, dim*2, H, W)

        # 细化
        out_dec_level1 = self.refinement(out_dec_level1)           # (B, dim*2, H, W)

        # 输出 + 全局残差连接（假设输入与输出空间对齐，例如去噪、增强等任务）
        out = self.output(out_dec_level1) + inp_img                # (B, out_channels, H, W)

        # 根据是否提供 label 决定返回结果或损失
        if label is None:
            return out
        else:
            return self.loss_fun(out, label)






