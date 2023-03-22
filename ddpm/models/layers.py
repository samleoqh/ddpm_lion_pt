import torch
from torch import nn
from torch.nn import Parameter, init

from ddpm.models.pos_embs.rotary import Rotary2D, apply_rotary_position_embeddings
from ddpm.utils import default

import math

__all__ = ["GroupNormCustom", "ResidualBlock", "Downsample", "Upsample",
           "AttentionBlock", "QKVAttention"]


class GroupNormCustom(nn.Module):

    def __init__(self, n_groups, num_channels, eps=1e-6, affine=True):
        super().__init__()
        self.n_groups = n_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = Parameter(torch.empty(num_channels))
            self.bias = Parameter(torch.empty(num_channels))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(b, h, w, self.n_groups, c // self.n_groups)  # (b, h, w, g, f) s.t. c = g * f
        var, mean = torch.var_mean(x, dim=[1, 2, 3], keepdim=True)
        # norm_x = (x - mean) * torch.rsqrt(var + self.eps)  # (b, h, w, g, f)
        norm_x = x.sub(mean).mul(torch.rsqrt(var + self.eps))
        norm_x = norm_x.flatten(start_dim=-2)  # (b, h, w, c)
        if self.affine:
            # norm_x = norm_x * self.weight + self.bias
            norm_x = norm_x.mul(self.weight[None, None, None]).add(self.bias[None, None, None])
        return norm_x.permute(0, 3, 1, 2)


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="bilinear"),
        nn.Conv2d(dim, default(dim_out, dim), (3, 3), padding=1, bias=False),
        nn.SiLU(inplace=True),
        GroupNormCustom(32, default(dim_out, dim))
    )


def Downsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Conv2d(dim, default(dim_out, dim), (3, 3), padding=1, bias=False),
        nn.SiLU(inplace=True),
        GroupNormCustom(32, default(dim_out, dim)),
        nn.AvgPool2d((2, 2), (2, 2))
    )


class ResidualBlock(nn.Module):

    def __init__(self, in_c, out_c, emb_dim, n_groups=32, with_time_emb=True):
        super().__init__()
        self.with_time_emb = with_time_emb
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=(1, 1), bias=False)
        if with_time_emb:
            self.dense = nn.Linear(emb_dim, out_c, bias=False)
        self.fn1 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=(3, 3), padding=1, bias=False),
            nn.SiLU(inplace=True),
        )
        self.fn2 = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=(3, 3), padding=1, bias=False),
            nn.SiLU(inplace=True),
        )
        self.out_c = out_c
        self.pre_norm = GroupNormCustom(n_groups, out_c)
        self.post_norm = GroupNormCustom(n_groups, out_c)

    def forward(self, x, t):
        if x.shape[1] == self.out_c:
            xi = x
        else:
            xi = x = self.conv(x)
        x = self.pre_norm(x)
        x = self.fn1(x)
        if self.with_time_emb:
            x = x + self.dense(t).unsqueeze(-1).unsqueeze(-1)
        x = self.post_norm(x)
        x = self.fn2(x)
        return xi + x


class QKVAttention(nn.Module):

    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv: torch.Tensor):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, 1)  # 3 x (B, C, -1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum("bct,bcs->bts",
                              (q * scale).reshape(bs * self.n_heads, ch, length),
                              (k * scale).reshape(bs * self.n_heads, ch, length))
        # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.to(torch.float16), -1).to(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)


class AttentionBlock(nn.Module):

    def __init__(self, channels: int, num_heads: int, num_head_channels: int):
        super().__init__()
        self.rotary = Rotary2D(channels)
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0, f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels

        self.norm = GroupNormCustom(min(32, channels // 4), channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.attn = QKVAttention(num_heads)
        self.proj_out = nn.Conv1d(channels, channels, 1)

        self.zero_init_weights()

    @torch.no_grad()
    def zero_init_weights(self):
        for p in self.proj_out.parameters():
            p.zero_()

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape

        x = self.norm(x)
        xi = x.view(B, C, -1)
        qkv = self.qkv(xi)  # (B, 3 x C, -1)
        # ---------------- Apply rotary 2d position embedding -----------------
        q_, k_, v_ = qkv.permute(0, 2, 1).chunk(3, 2)  # (B, -1, C)
        q_, k_ = apply_rotary_position_embeddings(self.rotary.forward(x), q_, k_)
        qkv = torch.cat((q_, k_, v_), 2).permute(0, 2, 1)  # (B, 3 x C, -1)
        # ----------------------------------------------------------------------
        h = self.attn(qkv)
        h = self.proj_out(h)
        return (xi + h).reshape(B, C, H, W)
