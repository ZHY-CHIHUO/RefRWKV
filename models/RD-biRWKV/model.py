# Copyright (c) Shanghai AI Lab. All rights reserved.
import math, os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F 
from einops import rearrange
from torch.utils.cpp_extension import load

wkv_cuda = load(
    name="bi_wkv",
    sources=["./cuda/bi_wkv.cpp", "./cuda/bi_wkv_kernel.cu"],
    verbose=True,
    extra_cuda_cflags=[
        "-res-usage",
        "--maxrregcount 60",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        "-gencode arch=compute_120,code=sm_120",
    ],
)


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
