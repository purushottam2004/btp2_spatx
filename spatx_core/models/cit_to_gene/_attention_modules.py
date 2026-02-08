from typing import Optional
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Softmax, Conv2d, Parameter, Module


from einops.layers.torch import Rearrange

class _CAM_Module(Module):
    def __init__(self, in_dim : int, dim : int, num_heads : int, qk_scale : Optional[float] = None, C_lambda : float = 1e-4, attn_drop : float = 0., proj_drop : float = 0.):
        super(_CAM_Module, self).__init__() #type: ignore
        self.chanel_in = in_dim
        self.softmax = Softmax(dim=-1)
        self.c_lambda = C_lambda
        self.activaton = nn.Sigmoid()
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Sequential(nn.Conv2d(dim//12, dim, kernel_size=3, padding=1, stride=1, bias=False, groups=dim//12), nn.GELU())
        self.proj_drop = nn.Dropout(proj_drop)

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//12, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//12, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//12, kernel_size=1)

        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x : Tensor, mask : Tensor | None =None):
        m_batchsize, N, C = x.size()
        height = int(N ** .5)
        width = int(N ** .5)

        x = x.view(m_batchsize, C, height, width)
        proj_query : Tensor = self.query_conv(x).view(m_batchsize, C//12, -1)
        proj_key : Tensor = self.key_conv(x).view(m_batchsize, C//12, -1).permute(0, 2, 1)
        proj_value : Tensor = self.value_conv(x).view(m_batchsize, C//12, -1)

        q = proj_query * self.scale
        attn = (q @ proj_key)

        if mask is not None:
            nW = mask.shape[0]  # num_windows
            attn = attn.view(m_batchsize // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ proj_value).reshape(m_batchsize, C//12, height, width)
        x = self.proj(x)
        x = x.reshape(m_batchsize, C, N).transpose(1, 2)
        x = self.proj_drop(x)

        out = self.gamma * x + x
        return out

class _PAM_Module(Module):
    def __init__(self, in_dim : int, dim : int, num_heads : int, qk_scale : float | None =None, P_lambda : float =1e-4, attn_drop : float =0., proj_drop : float =0.):
        super(_PAM_Module, self).__init__() #type: ignore
        self.chanel_in = in_dim
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Sequential(nn.Conv2d(dim//12, dim, kernel_size=3, padding=1, stride=1, bias=False, groups=dim//12), nn.GELU())
        self.proj_drop = nn.Dropout(proj_drop)

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//12, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//12, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//12, kernel_size=1)
        self.softmax = Softmax(dim=-1)


        self.p_lambda = P_lambda
        self.activaton = nn.Sigmoid()
        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x : Tensor, mask : Tensor | None =None):
        m_batchsize, N, C = x.size()
        height = int(N ** .5)
        width = int(N ** .5)

        x = x.view(m_batchsize, C, height, width)
        proj_query : Tensor = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key : Tensor = self.key_conv(x).view(m_batchsize, -1, width*height)
        proj_value : Tensor = self.value_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)

        q = proj_query * self.scale
        attn = (q @ proj_key)

        if mask is not None:
            nW = mask.shape[0]  # num_windows
            attn = attn.view(m_batchsize // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ proj_value).reshape(m_batchsize, C//12, height, width)
        x = self.proj(x)
        x = x.reshape(m_batchsize, C, N).transpose(1, 2)
        x = self.proj_drop(x)

        out = self.gamma * x + x
        return out

class _CHAM_Module(Module):
    def __init__(self, in_dim : int, dim : int, num_heads : int, qk_scale : float | None =None, P_lambda : float =1e-4, attn_drop : float =0., proj_drop : float =0.):
        super(_CHAM_Module, self).__init__() #type: ignore
        self.chanel_in = in_dim

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Sequential(nn.Conv2d(dim//12, dim, kernel_size=3, padding=1, stride=1, bias=False, groups=dim//12), nn.GELU())
        self.proj_drop = nn.Dropout(proj_drop)

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//12, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//12, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//12, kernel_size=1)
        self.softmax = Softmax(dim=-1)

        self.p_lambda = P_lambda
        self.activaton = nn.Sigmoid()
        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x : Tensor, mask : Tensor | None =None):
        m_batchsize, N, C = x.size()
        height = int(N ** .5)
        width = int(N ** .5)

        x = x.view(m_batchsize, C, height, width)
        proj_query = self.query_conv(x).view(m_batchsize, C//12 * height, -1)
        proj_key = self.key_conv(x).view(m_batchsize, C//12 * height, -1).permute(0, 2, 1)
        proj_value = self.value_conv(x).view(m_batchsize, C//12 * height, -1)

        q = proj_query * self.scale
        attn = (q @ proj_key)

        if mask is not None:
            nW = mask.shape[0]  # num_windows
            attn = attn.view(m_batchsize // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ proj_value).reshape(m_batchsize, C//12, height, width)
        x = self.proj(x)
        x = x.reshape(m_batchsize, C, N).transpose(1, 2)
        x = self.proj_drop(x)

        out = self.gamma * x + x
        return out


class _CWAM_Module(Module):
    def __init__(self, in_dim : int, dim : int, num_heads : int, qk_scale : float | None =None, P_lambda : float =1e-4, attn_drop : float =0., proj_drop : float =0.):
        super(_CWAM_Module, self).__init__() #type: ignore
        self.chanel_in = in_dim
        self.num_heads = num_heads

        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, bias=False, groups=dim), nn.GELU())
        self.proj_drop = nn.Dropout(proj_drop)

        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//12, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//12, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=-1)

        self.p_lambda = P_lambda
        self.activaton = nn.Sigmoid()
        self.gamma = Parameter(torch.zeros(1))

    def forward(self, x : Tensor, mask : Tensor | None =None):
        m_batchsize, N, C = x.size()
        height = int(N ** .5)
        width = int(N ** .5)

        proj_query : Tensor = x.view(m_batchsize, C * width, -1)
        proj_key : Tensor = x.view(m_batchsize, C * width, -1).permute(0, 2, 1)
        proj_value : Tensor = x.view(m_batchsize, C * width, -1)

        q = proj_query * self.scale
        attn = (q @ proj_key)

        if mask is not None:
            nW = mask.shape[0]  # num_windows
            attn = attn.view(m_batchsize // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        x = (attn @ proj_value).reshape(m_batchsize, C, height, width)
        x = self.proj(x)
        x = x.reshape(m_batchsize, C, N).transpose(1, 2)
        x = self.proj_drop(x)

        out = self.gamma * x + x
        return out

class WindowAttention_ACAM(nn.Module):
    def __init__(self, dim : int, num_heads : int, qkv_bias : bool =True, qk_scale : float | None =None, attn_drop : float =0., proj_drop : float =0.):
        super().__init__() #type: ignore
        self.dim = dim

        self.C_C = _CAM_Module(self.dim, dim=dim, num_heads=num_heads)
        self.H_W = _PAM_Module(self.dim, dim=dim, num_heads=num_heads)
        self.C_H = _CHAM_Module(self.dim, dim=dim, num_heads=num_heads)
        self.C_W = _CWAM_Module(self.dim, dim=dim, num_heads=num_heads)

        self.gamma1 = Parameter(torch.zeros(1))
        self.gamma2 = Parameter(torch.zeros(1))
        self.gamma3 = Parameter(torch.ones(1) * 0.5)
        self.gamma4 = Parameter(torch.ones(1) * 0.5)

    def _build_projection(self, dim_in : int, kernel_size : int =3, stride : int =1, padding : int =1):
        proj = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size, padding=padding, stride=stride, bias=False, groups=dim_in),
            Rearrange('b c h w -> b (h w) c'),
            nn.LayerNorm(dim_in))
        return proj

    def forward(self, x : Tensor, mask : Tensor | None =None):
        x_out1 = self.C_C(x)

        x_out2 = self.H_W(x)

        x_out3 = self.C_H(x)

        x_out4 = self.C_W(x)

        x_out = (self.gamma1 * x_out1) + (self.gamma2 * x_out2) + (self.gamma3 * x_out3) + (self.gamma4 * x_out4)

        return x_out
