# Copyright (c) Shanghai AI Lab. All rights reserved.
import math, os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F 
from einops import rearrange
from torch.utils.cpp_extension import load
wkv_cuda = load(name="bi_wkv", sources=["./cuda/bi_wkv.cpp", "./cuda/bi_wkv_kernel.cu"],
                        verbose=True, extra_cuda_cflags=['-res-usage', '--maxrregcount 60', '--use_fast_math', '-O3', '-Xptxas -O3', '-gencode arch=compute_120,code=sm_120'])
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

class Restore_RWKV_Ref(nn.Module):
    def __init__(self,
        inp_channels=8,                # 单个低分图像的通道数
        out_channels=4,
        dim = 48,
        num_blocks = [4,6,6,8],
        num_refinement_blocks = 8,
        loss_fun = nn.L1Loss(),
        scale = 8
    ):
        super().__init__()
        self.scale = scale
        self.loss_fun = loss_fun

        # ---------- 1. 输入融合：拼接 lr1 与 lr2 ----------
        # 原 patch_embed 改为接受 2*inp_channels 通道
        self.lr_fuse = nn.Conv2d(inp_channels * 2, dim, 3, padding=1, bias=True)

        # ---------- 2. 参考图像多尺度提取 ----------
        self.ref_extractor = RefExtractor(out_channels, dim)

        # ---------- 3. 各层级 1×1 融合卷积 ----------
        self.fuse1 = nn.Conv2d(dim * 2,      dim,      1, bias=False)
        self.fuse2 = nn.Conv2d(dim * 4,      dim * 2,  1, bias=False)
        self.fuse3 = nn.Conv2d(dim * 8,      dim * 4,  1, bias=False)
        self.fuse4 = nn.Conv2d(dim * 16,     dim * 8,  1, bias=False)

        # ---------- 4. U‑Net 编码器（尺寸与通道与原模型一致） ----------
        self.encoder_level1 = nn.Sequential(*[Block(n_embd=dim) for _ in range(num_blocks[0])])
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[Block(n_embd=int(dim*2)) for _ in range(num_blocks[1])])
        self.down2_3 = Downsample(int(dim*2))
        self.encoder_level3 = nn.Sequential(*[Block(n_embd=int(dim*4)) for _ in range(num_blocks[2])])
        self.down3_4 = Downsample(int(dim*4))
        self.latent     = nn.Sequential(*[Block(n_embd=int(dim*8)) for _ in range(num_blocks[3])])

        # ---------- 5. U‑Net 解码器 ----------
        self.up4_3 = Upsample(int(dim*8))
        self.reduce_chan_level3 = nn.Conv2d(int(dim*8), int(dim*4), 1, bias=True)
        self.decoder_level3 = nn.Sequential(*[Block(n_embd=int(dim*4)) for _ in range(num_blocks[2])])
        self.up3_2 = Upsample(int(dim*4))
        self.reduce_chan_level2 = nn.Conv2d(int(dim*4), int(dim*2), 1, bias=True)
        self.decoder_level2 = nn.Sequential(*[Block(n_embd=int(dim*2)) for _ in range(num_blocks[1])])
        self.up2_1 = Upsample(int(dim*2))
        self.decoder_level1 = nn.Sequential(*[Block(n_embd=int(dim*2)) for _ in range(num_blocks[0])])
        self.refinement      = nn.Sequential(*[Block(n_embd=int(dim*2)) for _ in range(num_refinement_blocks)])

        # ---------- 6. 超分上采样与输出 ----------
        up_hr_layers = []
        for _ in range(int(math.log2(scale))):
            up_hr_layers += [
                nn.Conv2d(int(dim*2), int(dim*2)*4, 3, padding=1, bias=False),
                nn.PixelShuffle(2)
            ]
        self.up_hr = nn.Sequential(*up_hr_layers)
        self.output = nn.Sequential(nn.Conv2d(int(dim*2), out_channels, 3, padding=1, bias=True), nn.Sigmoid())

    def forward(self, lr1, hr1, lr2, label=None):
        # a) 拼接低分输入
        lr_cat = torch.cat([lr1, lr2], dim=1)          # (B, 16, 32, 32)
        fea = self.lr_fuse(lr_cat)                     # (B, dim, 32, 32)

        # b) 多尺度参考特征
        f32, f16, f8, f4 = self.ref_extractor(hr1)
        # f32: (B,dim,32)  f16: (B,2dim,16)  f8: (B,4dim,8)  f4: (B,8dim,4)

        # c) 编码器：逐级融合参考特征
        e1 = self.encoder_level1(self.fuse1(torch.cat([fea, f32], dim=1)))
        e2 = self.encoder_level2(self.fuse2(torch.cat([self.down1_2(e1), f16], dim=1)))
        e3 = self.encoder_level3(self.fuse3(torch.cat([self.down2_3(e2), f8], dim=1)))
        l  = self.latent(self.fuse4(torch.cat([self.down3_4(e3), f4], dim=1)))

        # d) 解码器（跳跃连接使用已融合的 e1, e2, e3）
        d3 = self.decoder_level3(self.reduce_chan_level3(torch.cat([self.up4_3(l), e3], 1)))
        d2 = self.decoder_level2(self.reduce_chan_level2(torch.cat([self.up3_2(d3), e2], 1)))
        d1 = self.decoder_level1(torch.cat([self.up2_1(d2), e1], 1))

        # e) 精炼与输出
        d1 = self.refinement(d1)          # 原模型此处有两次 refinement，保留一次即可
        hr_feat = self.up_hr(d1)
        out_hr = self.output(hr_feat)     # (B, out_channels, 256, 256)

        if label is None:
            return out_hr
        else:
            return self.loss_fun(out_hr, label)


class RefExtractor(nn.Module):
    """从 256×256 参考图像提取四层特征，下采样至 32,16,8,4"""
    def __init__(self, in_ch, dim):
        super().__init__()
        # 256 -> 128 -> 64 -> 32
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, stride=2, padding=1),
        )
        self.to_f16 = nn.Conv2d(dim, dim*2, 3, stride=2, padding=1)
        self.to_f8  = nn.Conv2d(dim*2, dim*4, 3, stride=2, padding=1)
        self.to_f4  = nn.Conv2d(dim*4, dim*8, 3, stride=2, padding=1)

    def forward(self, hr):
        f32 = self.stem(hr)        # (B, dim, 32, 32)
        f16 = F.relu(self.to_f16(f32))
        f8  = F.relu(self.to_f8(f16))
        f4  = F.relu(self.to_f4(f8))
        return f32, f16, f8, f4


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    lr1 = torch.randn(4, 8, 32, 32).cuda()   # 参考低分
    hr1 = torch.randn(4, 4, 256, 256).cuda() # 参考高分
    lr2 = torch.randn(4, 8, 32, 32).cuda()   # 待超分
    hr2 = torch.randn(4, 4, 256, 256).cuda() # 参考高分

    model = Restore_RWKV_Ref().cuda()
    loss = model(lr1, hr1, lr2, hr2)
    print(loss)
    pred_hr2 = model(lr1, hr1, lr2)
    print(pred_hr2.shape)  # (4, 4, 256, 256)
    model = Restore_RWKV_Ref().cuda()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数：{total_params/1e6:.2f} M")
