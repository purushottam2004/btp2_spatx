"""
CiT-Net-Tiny
            stage1  stage2  stage3  stage4
    size    56x56   28x28   14x14   7x7
Unet dim    96      192     384     768
Swin dim    96      192     384     768
     head   3       6       12      24
     num    2       2       6       2
"""
from typing import Type, List

import torch
import math
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.layers.drop import DropPath
from torch import Tensor
from torch.nn import Module

from ._attention_modules import WindowAttention_ACAM
from ._DDConv import DDConv
from ._utils import window_partition, window_reverse

class PatchEmbed(nn.Module):
    def __init__(self, img_size : int =224, patch_size : int =4, in_chans : int =3, embed_dim : int =96, norm_layer : Type[Module] | None = None):
        super().__init__() #type: ignore
        self.patch_size = (patch_size, patch_size)
        self.patches_resolution = (
            img_size // self.patch_size[0],
            img_size // self.patch_size[1]
        )
        self.embed_dim = embed_dim
        self.in_chans = in_chans
        self.conv2patch = nn.Sequential(
            nn.Conv2d(3, embed_dim, kernel_size=4, stride=4),
            nn.GELU(),
            nn.BatchNorm2d(embed_dim)
        )

    def forward(self, x : Tensor):
        # # FIXME look at relaxing size constraints
        x = self.conv2patch(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        flops += Ho * Wo * self.embed_dim
        return flops

class oneXone_conv(nn.Module):
    def __init__(self, in_features : int, hidden_features : int | None =None, out_features : int | None =None, act_layer : Type[nn.Module] =nn.GELU, drop : float =0.):
        super(oneXone_conv, self).__init__() #type: ignore
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.Conv1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(hidden_features)
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(out_features)
        )
        self.drop = nn.Dropout(drop)
    def forward(self, x : Tensor):
        x = self.Conv1(x)
        x = self.drop(x)
        x = self.Conv2(x)
        x = self.drop(x)
        return x

class GhostModule(nn.Module):
    def __init__(self, inp : int, oup : int | None =None, kernel_size : int =1, ratio : int =2, dw_size : int =3, stride : int =1, relu : bool =True):
        super(GhostModule, self).__init__() #type: ignore
        oup = oup or inp
        init_channels = math.ceil(oup // ratio)
        new_channels = init_channels*(ratio-1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x : Tensor):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out

class GhostModule_Up(nn.Module):
    def __init__(self, inp : int, oup : int | None =None, kernel_size : int =1, ratio : int =2, dw_size : int =3, stride : int =1, relu : bool =True):
        super(GhostModule_Up, self).__init__() #type: ignore
        oup = oup or inp
        init_channels = inp
        new_channels = init_channels

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x : Tensor):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out

class SwinTransformerBlock(nn.Module):
    def __init__(self, dim : int, input_resolution : tuple[int, int], num_heads : int, window_size : int =7, shift_size : int =0,
                 mlp_ratio : float =2., qkv_bias : bool =True, qk_scale : float | None =None, drop : float =0., attn_drop : float =0., drop_path : List[float] =[0.],
                 act_layer : Type[nn.Module] =nn.GELU, norm_layer : Type[nn.Module] =nn.LayerNorm):
        super().__init__() #type: ignore

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio


        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer([dim, input_resolution[0], input_resolution[1]])

        self.attn = WindowAttention_ACAM(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path[0]) if drop_path[0] > 0. else nn.Identity()

        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.drop_path = DropPath(drop_path[0]) if drop_path[0] > 0. else nn.Identity()
        self.mlp = GhostModule(inp=dim)

    def forward(self, x : Tensor):
        B, C, H, W = x.shape

        shortcut1 = x
        x = x.view(B, H, W, C)
        x = self.norm1(x)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, C, H, W)
        x = shortcut1 + self.drop_path(x)
        shortcut2 = x

        x = self.norm2(x)

        x = shortcut2 + self.drop_path(self.mlp(x))
        return x

class PatchMerging(nn.Module):
    def __init__(self, input_resolution : tuple[int, int], dim : int, norm_layer : Type[nn.Module] =nn.LayerNorm):
        super().__init__() #type: ignore
        self.input_resolution = input_resolution
        h, w = input_resolution
        h = int(h/2)
        w = int(w/2)
        self.dim = dim
        self.norm = norm_layer([4*dim, h, w])
        self.reduction = GhostModule(inp=4 * dim, oup=2 * dim, ratio=4)

    def forward(self, x : Tensor):
        x0 = x[:, :, 0::2, 0::2]  # B C H/2 W/2
        x1 = x[:, :, 1::2, 0::2]  # B C H/2 W/2
        x2 = x[:, :, 0::2, 1::2]  # B C H/2 W/2
        x3 = x[:, :, 1::2, 1::2]  # B C H/2 W/2
        x = torch.cat([x0, x1, x2, x3], 1)  # B 4*C H/2 W/2

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class BasicLayer(nn.Module):
    def __init__(self, dim : int , input_resolution : tuple[int, int], depth, num_heads : int , window_size: int,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path : List[float] = [0.], norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x : Tensor):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

class PatchExpand(nn.Module):
    def __init__(self, input_resolution : tuple[int, int], dim : int, dim_scale : int =2, norm_layer : Type[nn.Module] =nn.LayerNorm):
        super().__init__() #type: ignore
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = GhostModule_Up(inp=dim) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer([dim // dim_scale, input_resolution[0]*2, input_resolution[1]*2])

    def forward(self, x: Tensor):
        _, C, _, _  = x.shape
        x = self.expand(x)
        x = rearrange(x, 'b (p1 p2 c) h w -> b c (h p1) (w p2)', p1=2, p2=2, c=C // 2)
        x = self.norm(x)

        return x

class BasicLayer_up(nn.Module):
    def __init__(self, dim , input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path : List[float] =[0.], norm_layer : Type[Module] = nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        if upsample is not None:
            self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x : Tensor):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x

class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution : tuple[int, int], dim : int, dim_scale : int =4, norm_layer : Type[nn.Module] =nn.LayerNorm):
        super().__init__() #type: ignore
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = oneXone_conv(in_features = dim, out_features = 16 * dim) if dim_scale == 2 else nn.Identity()
        self.output_dim = dim
        self.norm = norm_layer([6, input_resolution[0]*4, input_resolution[1]*4])

    def forward(self, x: Tensor):
        _, C, _, _ = x.shape
        x = self.expand(x)
        x = rearrange(x, 'b (p1 p2 c) h w -> b c (h p1) (w p2)', p1=self.dim_scale, p2=self.dim_scale, c=C // (self.dim_scale ** 2))
        x = self.norm(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, ch_in : int , ch_out : int ):
        super(ConvBlock, self).__init__() #type: ignore
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x : Tensor):
        x = self.conv(x)
        return x


class ConvBlockDDConv(nn.Module):
    def __init__(self, ch_in : int , ch_out : int):
        super(ConvBlockDDConv, self).__init__() #type: ignore
        self.conv = nn.Sequential(
            DDConv(ch_in, ch_out, bias = True, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),

            DDConv(ch_out, ch_out, bias= True, kernel_size=3, stride=1, padding=1),

            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x : Tensor):
        x = self.conv(x)
        return x

class UpConvDDConv(nn.Module):
    def __init__(self, ch_in : int, ch_out :int):
        super(UpConvDDConv, self).__init__() #type: ignore
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            DDConv(ch_in, ch_out,kernel_size=1, bias = True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x : Tensor):
        x = self.up(x)
        return x

class ConvMixerLayer(nn.Module):
    def __init__(self, dim : int , kernel_size : int =9):
        super().__init__() #type: ignore
        self.Resnet = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, groups=dim, padding=4),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        self.Conv_1x1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
    def forward(self, x : Tensor):
        x = x + self.Resnet(x)
        x = self.Conv_1x1(x)
        return x


class ConvMixer(nn.Module):
    def __init__(self, dim : int =512, depth : int =1, kernel_size : int =9, patch_size : int =4, n_classes : int =1000):
        super().__init__() #type: ignore
        self.conv2d1 = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        )
        self._ConvMixer_blocks = nn.ModuleList([])

        for _ in range(depth):
            self._ConvMixer_blocks.append(ConvMixerLayer(dim=dim, kernel_size=kernel_size))

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(dim, n_classes)
        )

    def forward(self, x : Tensor):
        x = self.conv2d1(x)

        for _ConvMixer_block in self._ConvMixer_blocks:
            x = _ConvMixer_block(x)

        x = x
        return x


