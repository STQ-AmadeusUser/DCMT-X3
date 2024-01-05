import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from timm.models.layers import to_2tuple, trunc_normal_, DropPath


class BN_Activ_Conv(nn.Module):
    def __init__(self, in_channels, activation, out_channels, kernel_size, stride=(1, 1), dilation=(1, 1), groups=1):
        super(BN_Activ_Conv, self).__init__()
        self.BN = nn.BatchNorm2d(out_channels)
        self.Activation = activation
        padding = [int((dilation[j] * (kernel_size[j] - 1) - stride[j] + 1) / 2) for j in range(2)]  # Same padding
        self.Conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups=groups, bias=False)

    def forward(self, img):
        img = self.BN(img)
        img = self.Activation(img)
        img = self.Conv(img)
        return img


class DepthWise_Conv(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_merge = BN_Activ_Conv(channels, nn.GELU(), channels, (3, 3), groups=channels)

    def forward(self, img):
        img = self.conv_merge(img)
        return img


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., bias=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)
        trunc_normal_(self.embeddings_table, std=.02)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat)
        embeddings = self.embeddings_table[final_mat]

        return embeddings


class StripMLP_Block(nn.Module):
    def __init__(self, channels, H, W):
        super().__init__()
        assert W == H
        self.channels = channels
        self.activation = nn.GELU()
        self.BN = nn.BatchNorm2d(channels//2)

        if channels % 80 == 0:
            patch = 2
        else:
            patch = 4

        self.ratio = 1; self.C = int(channels *0.5/ patch); self.chan = self.ratio * self.C

        self.proj_h = nn.Conv2d(H*self.C, self.chan*H, (1, 3), stride=1, padding=(0, 1), groups=self.C,bias=True)
        self.proj_w = nn.Conv2d(self.C*W, self.chan*W, (1, 3), stride=1, padding=(0, 1), groups=self.C, bias=True)

        self.fuse_h = nn.Conv2d(channels, channels//2, (1,1), (1,1), bias=False)
        self.fuse_w = nn.Conv2d(channels, channels//2, (1,1), (1,1), bias=False)

        self.mlp=nn.Sequential(nn.Conv2d(channels, channels, 1, 1,bias=True),nn.BatchNorm2d(channels),nn.GELU())

        dim = channels // 2

        self.fc_h = nn.Conv2d(dim, dim, (3,7), stride=1, padding=(1,7//2), groups=dim, bias=False)
        self.fc_w = nn.Conv2d(dim, dim, (7,3), stride=1, padding=(7//2,1), groups=dim, bias=False)

        self.reweight = Mlp(dim, dim // 2, dim * 3)

        self.fuse = nn.Conv2d(channels, channels, (1,1), (1,1), bias=False)

        self.relate_pos_h = RelativePosition(channels//2, H)
        self.relate_pos_w = RelativePosition(channels//2, W)

    def forward(self, x):
        N, C, H, W = x.shape

        x = self.mlp(x)

        x_1 = x[:, :C//2, :, :]
        x_2 = x[:, C//2:, :, :]

        x_1 = self.strip_mlp(x_1)

        # for x_2
        x_w = self.fc_h(x_2)
        x_h = self.fc_w(x_2)
        att = F.adaptive_avg_pool2d(x_h + x_w + x_2, output_size=1)
        att = self.reweight(att).reshape(N, C//2, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        x_2 = x_h * att[0] + x_w * att[1] + x_2 * att[2]

        x = self.fuse(torch.cat([x_1, x_2], dim=1))

        return x

    def strip_mlp(self, x):
        N, C, H, W = x.shape

        pos_h = self.relate_pos_h(H, W).unsqueeze(0).permute(0, 3, 1, 2)
        pos_w = self.relate_pos_w(H, W).unsqueeze(0).permute(0, 3, 1, 2)

        C1 = int(C/self.C)

        x_h = x + pos_h
        x_h = x_h.view(N, C1, self.C, H, W)     # N C1 C2 H W

        x_h = x_h.permute(0, 1, 3, 2, 4).contiguous().view(N, C1, H, self.C*W)  # N C1 H WC2

        x_h = self.proj_h(x_h.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x_h = x_h.view(N, C1, H, self.C, W).permute(0, 1, 3, 2, 4).contiguous().view(N, C, H, W) # N C1 C2 H W

        x_h = self.fuse_h(torch.cat([x_h, x], dim=1))
        x_h = self.activation(self.BN(x_h)) + pos_w

        x_w = self.proj_w(x_h.view(N, C1, H*self.C, W).permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        x_w = x_w.contiguous().view(N, C1, self.C, H, W).view(N, C, H, W)

        x = self.fuse_w(torch.cat([x, x_w], dim=1))

        return x


class TokenMixing(nn.Module):
    r""" Token mixing of Strip MLP

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, C, H, W):
        super().__init__()
        self.smlp_block = StripMLP_Block(C, H, W)
        self.dwsc = DepthWise_Conv(C)

    def forward(self, x):
        x = self.dwsc(x)
        x = self.smlp_block(x)

        return x


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ChannelMixing(nn.Module):

    def __init__(self, in_channel, out_channel, alpha, use_dropout=False, drop_rate=0):
        super().__init__()

        self.use_dropout = use_dropout

        self.conv_77 = nn.Conv2d(in_channel, in_channel, 7, 1, 3, groups=in_channel, bias=False)
        self.layer_norm = nn.LayerNorm(in_channel)
        self.fc1 = nn.Linear(in_channel, alpha * in_channel)
        self.activation = nn.GELU()
        self.drop = nn.Dropout(drop_rate)
        self.fc2 = nn.Linear(alpha * in_channel, out_channel)

        self.grn = GRN(alpha * in_channel)

    def forward(self, x):
        N, C, H, W = x.shape

        x = self.conv_77(x)
        x = x.permute(0, 2, 3, 1)
        x = self.layer_norm(x)

        x = self.fc1(x)
        x = self.activation(x)
        x = self.grn(x)

        x = self.fc2(x)

        x = x.permute(0, 3, 1, 2)

        return x


class WaveMLP_Strip(nn.Module):
    def __init__(self, in_channel, out_channel, HW, t, use_dropout=False, drop_rate=0., affine=True, track_running_stats=True):
        super().__init__()

        self.token_mixing = TokenMixing(in_channel, HW, HW)
        self.channel_mixing = ChannelMixing(in_channel, out_channel,
                                            alpha=t, use_dropout=use_dropout, drop_rate=drop_rate)

        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.token_mixing(x))
        x = x + self.drop_path(self.channel_mixing(x))

        return x


class ResNetBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, t=3, dw=False, stride=1, dilation=1, affine=True, track_running_stats=True):
        super(ResNetBlock, self).__init__()
        self.embed = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0)

        self.conv1 = nn.Conv2d(planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine, track_running_stats=track_running_stats)
        if dw:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride,
                                   padding=padding, bias=False, dilation=dilation, groups=planes)
        else:
            self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, stride=stride,
                                   padding=padding, bias=True, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine, track_running_stats=track_running_stats)
        self.conv3 = nn.Conv2d(planes, planes * t, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * t, affine=affine, track_running_stats=track_running_stats)
        self.act = nn.ReLU(inplace=True)
        self.proj = nn.Sequential(
            nn.Conv2d(planes * t, planes, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(planes, affine=affine, track_running_stats=track_running_stats),
        )

    def forward(self, x):
        x = self.embed(x)
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.proj(out)
        out += residual

        out = self.act(out)

        return out


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        return x


class ViTBlock(nn.Module):

    def __init__(self, in_dim, dim, num_heads=4, t=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm):
        super(ViTBlock, self).__init__()
        self.embed = nn.Conv2d(in_dim, dim, kernel_size=1, stride=1, padding=0)

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * t)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop, bias=True)

    def forward(self, x):
        x = self.embed(x)
        _, C, H, W = x.size()

        x = x.flatten(2).permute(0, 2, 1).contiguous()
        x = x + self.drop_path(self.attn(self.norm1(x)))

        res = x.permute(0, 2, 1).contiguous().view(-1, C, H, W)
        x = self.norm2(x)
        x = x.permute(0, 2, 1).contiguous().view(-1, C, H, W)
        x = res + self.drop_path(self.mlp(x))

        return x
