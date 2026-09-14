import torch
import torch.nn as nn
import torch.nn.functional as F
from .vss_blocks import dual_vss_block
from einops import rearrange

class PatchExpand(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(
            dim, 4*dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim*2 // dim_scale)

    def forward(self, x, y):
        x = x.permute(0,2,3,1) 
        x = self.expand(x) 
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x= self.norm(x)
        x = x.permute(0,3,1,2)+y 
        return x

class PatchExpand_2(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(
            dim, 4*dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim*2 // dim_scale)

    def forward(self, x):
        x = x.permute(0,2,3,1) 
        x = self.expand(x) 
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x= self.norm(x)
        x = x.permute(0,3,1,2) 
        return x

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):

        x = self.expand(x)
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x= self.norm(x)
        return x

class Decoder(nn.Module):
    def __init__(self, encoder_dims, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs):
        super(Decoder, self).__init__()
        M = 36#64#48#36
        vss = dict(
            channel_first=channel_first,
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            kwargs=kwargs,
        )
        # Spatio-spectral-temporal scans.  Leading 1x1 matches the official
        # in_channels; the dual VSS keeps the published hidden dims.
        self.st_block_41 = dual_vss_block(M, 4, in_channels=encoder_dims[-1] * 3, **vss)
        self.st_block_42 = dual_vss_block(M, 12, in_channels=encoder_dims[-1], **vss)
        self.st_block_43 = dual_vss_block(M, 12, in_channels=encoder_dims[-1], **vss)
        self.st_block_31 = dual_vss_block(M, 8, in_channels=encoder_dims[-2] * 3, **vss)
        self.st_block_32 = dual_vss_block(M, 24, in_channels=encoder_dims[-2], **vss)
        self.st_block_33 = dual_vss_block(M, 24, in_channels=encoder_dims[-2], **vss)
        self.st_block_21 = dual_vss_block(M, 16, in_channels=encoder_dims[-3] * 3, **vss)
        self.st_block_22 = dual_vss_block(M, 48, in_channels=encoder_dims[-3], **vss)
        self.st_block_23 = dual_vss_block(M, 48, in_channels=encoder_dims[-3], **vss)
        self.st_block_11 = dual_vss_block(M, 32, in_channels=encoder_dims[-4] * 3, **vss)
        self.st_block_12 = dual_vss_block(M, 96, in_channels=encoder_dims[-4], **vss)
        self.st_block_13 = dual_vss_block(M, 96, in_channels=encoder_dims[-4], **vss)
        self.st_block_1 = dual_vss_block(M, 64, in_channels=M, **vss)
        N = M*7
        # Fuse layer  
        self.fuse_layer_4 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=N, out_channels=M),
                                          nn.BatchNorm2d(M), nn.ReLU())
        self.fuse_layer_3 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=N, out_channels=M),
                                          nn.BatchNorm2d(M), nn.ReLU())
        self.fuse_layer_2 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=N, out_channels=M),
                                          nn.BatchNorm2d(M), nn.ReLU())
        self.fuse_layer_1 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=N, out_channels=M),
                                          nn.BatchNorm2d(M), nn.ReLU())
        self.fuse_layer_0 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=M, out_channels=M),
                                          nn.BatchNorm2d(M), nn.ReLU())

        # Smooth layer
        self.smooth_layer_3 = ResBlock(in_channels=M, out_channels=M, stride=1) 
        self.smooth_layer_2 = ResBlock(in_channels=M, out_channels=M, stride=1) 
        self.smooth_layer_1 = ResBlock(in_channels=M, out_channels=M, stride=1)
        self.smooth_layer_0 = ResBlock(in_channels=M, out_channels=M, stride=1)

        # get down layer
        self.down_layer = CNNBlock(in_channels=M, out_channels=int(kwargs.get('in_chans', 4)), stride=1)
        self.subpixel_conv = nn.ConvTranspose2d(M, M, kernel_size=4, stride=2, padding=1) 
        self.pachexp = PatchExpand(M)
        self.pachexp_2 = PatchExpand_2(M)
    
    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, coarse0_fea, coarse1_fea, fine0_fea, def_device):

        coarse0_fea_1, coarse0_fea_2, coarse0_fea_3, coarse0_fea_4 = [x - y for x, y in zip(fine0_fea, coarse0_fea)]
        coarse1_fea_1, coarse1_fea_2, coarse1_fea_3, coarse1_fea_4 = [x - y for x, y in zip(coarse1_fea, coarse0_fea)]
        fine0_fea_1, fine0_fea_2, fine0_fea_3, fine0_fea_4 = coarse1_fea
        '''
            Stage I
        '''
        p41 = self.st_block_41(torch.cat([coarse0_fea_4, fine0_fea_4, coarse1_fea_4], dim=1))
        B, C, H, W = fine0_fea_4.size()
        ct_tensor_42 = torch.empty(B, C, H, 3*W).to(def_device) 
        # Fill in odd columns with A and even columns with B
        ct_tensor_42[:, :, :, ::3] = coarse0_fea_4  # Odd columns
        ct_tensor_42[:, :, :, 1::3] = fine0_fea_4  # Even columns
        ct_tensor_42[:, :, :, 2::3] = coarse1_fea_4 # Even columns
        p42 = self.st_block_42(ct_tensor_42)

        ct_tensor_43 = torch.empty(B, C, H, 3*W).to(def_device)
        ct_tensor_43[:, :, :, 0:W] = coarse0_fea_4
        ct_tensor_43[:, :, :, W:2*W] = coarse1_fea_4
        ct_tensor_43[:, :, :, 2*W:3*W] = fine0_fea_4 #8*768*8*24

        p43 = self.st_block_43(ct_tensor_43) #8*64*8*24
        p4 = self.fuse_layer_4(torch.cat([p41, p42[:, :, :, ::3], p42[:, :, :, 1::3], p42[:, :, :, 2::3],
                    p43[:, :, :, 0:W], p43[:, :, :, W:2*W], p43[:, :, :, 2*W:3*W]], dim=1))
        # '''
        #     Stage II
        # '''
        B, C, H, W = fine0_fea_3.size()
        p31 = self.st_block_31(torch.cat([coarse0_fea_3, fine0_fea_3, coarse1_fea_3], dim=1))
        # Create an empty tensor of the correct shape (B, C, H, 2*W)
        ct_tensor_32 = torch.empty(B, C, H, 3*W).to(def_device)
        # Fill in odd columns with A and even columns with B
        ct_tensor_32[:, :, :, ::3] = coarse0_fea_3  # Odd columns
        ct_tensor_32[:, :, :, 1::3] = coarse1_fea_3  # Even columns
        ct_tensor_32[:, :, :, 2::3] = fine0_fea_3  # Even columns
        p32 = self.st_block_32(ct_tensor_32)

        ct_tensor_33 = torch.empty(B, C, H, 3*W).to(def_device)
        ct_tensor_33[:, :, :, 0:W] = coarse0_fea_3
        ct_tensor_33[:, :, :, W:2*W] = coarse1_fea_3
        ct_tensor_33[:, :, :, 2*W:3*W] = fine0_fea_3
        p33 = self.st_block_33(ct_tensor_33)

        p3 = self.fuse_layer_3(torch.cat([p31, p32[:, :, :, ::3], p32[:, :, :, 1::3], p32[:, :, :, 2::3],
                    p33[:, :, :, 0:W], p33[:, :, :, W:2*W], p33[:, :, :, 2*W:3*W]], dim=1))
        p3 = self.pachexp(p4, p3)
        p3 = self.smooth_layer_3(p3)
        # '''
        #     Stage III
        # '''
        B, C, H, W = fine0_fea_2.size()
        p21 = self.st_block_21(torch.cat([coarse0_fea_2, fine0_fea_2, coarse1_fea_2], dim=1))
        ct_tensor_22 = torch.empty(B, C, H, 3*W).to(def_device)
        # Fill in odd columns with A and even columns with B
        ct_tensor_22[:, :, :, ::3] = coarse0_fea_2  # Odd columns
        ct_tensor_22[:, :, :, 1::3] = coarse1_fea_2  # Even columns
        ct_tensor_22[:, :, :, 2::3] = fine0_fea_2  # Even columns
        p22 = self.st_block_22(ct_tensor_22)

        ct_tensor_23 = torch.empty(B, C, H, 3*W).to(def_device)
        ct_tensor_23[:, :, :, 0:W] = coarse0_fea_2
        ct_tensor_23[:, :, :, W:2*W] = coarse1_fea_2
        ct_tensor_23[:, :, :, 2*W:3*W] = fine0_fea_2
        p23 = self.st_block_23(ct_tensor_23)

        p2 = self.fuse_layer_2(torch.cat([p21, p22[:, :, :, ::3], p22[:, :, :, 1::3], p22[:, :, :, 2::3],
                    p23[:, :, :, 0:W], p23[:, :, :, W:2*W], p23[:, :, :, 2*W:3*W]], dim=1))
        p2 = self.pachexp(p3, p2)
        p2 = self.smooth_layer_2(p2) 
        # '''
        #     Stage IV
        # '''
        B, C, H, W = fine0_fea_1.size()
        p11 = self.st_block_11(torch.cat([coarse0_fea_1, fine0_fea_1, coarse1_fea_1], dim=1))
        ct_tensor_12 = torch.empty(B, C, H, 3*W).to(def_device)
        ct_tensor_12[:, :, :, ::3] = coarse0_fea_1  
        ct_tensor_12[:, :, :, 1::3] = coarse1_fea_1  
        ct_tensor_12[:, :, :, 2::3] = fine0_fea_1  
        p12 = self.st_block_12(ct_tensor_12)

        ct_tensor_13 = torch.empty(B, C, H, 3*W).to(def_device)
        ct_tensor_13[:, :, :, 0:W] = coarse0_fea_1
        ct_tensor_13[:, :, :, W:2*W] = coarse1_fea_1
        ct_tensor_13[:, :, :, 2*W:3*W] = fine0_fea_1
        p13 = self.st_block_13(ct_tensor_13)

        p1 = self.fuse_layer_1(torch.cat([p11, p12[:, :, :, ::3], p12[:, :, :, 1::3], p12[:, :, :, 2::3],
                    p13[:, :, :, 0:W], p13[:, :, :, W:2*W], p13[:, :, :, 2*W:3*W]], dim=1))

        p1 = self.pachexp(p2, p1)
        p1 = self.smooth_layer_1(p1) 

        p1 = self.pachexp_2(p1)
        p1 = self.st_block_1(p1)
        p1 = self.fuse_layer_0(p1)
        p1 = self.smooth_layer_0(p1)
        p1 = self.pachexp_2(p1)
        p1 = self.down_layer(p1)
        return p1

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out
    
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(CNNBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv1(x)
        out = torch.tanh(out)
        return out
