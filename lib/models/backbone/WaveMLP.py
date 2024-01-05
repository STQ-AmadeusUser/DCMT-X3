import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import DropPath
from timm.models.layers.helpers import to_2tuple
import torch.nn.functional as F


# def MyNorm(dim, affine, track_running_stats):
#     return nn.BatchNorm2d(dim, affine=affine, track_running_stats=track_running_stats)
def MyNorm(dim):
    return nn.GroupNorm(1, dim)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x   


class PATM(nn.Module):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., mode='fc'):
        super().__init__()

        self.fc_h = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)
        self.fc_w = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)
        self.fc_c = nn.Conv2d(dim, dim, 1, 1, bias=qkv_bias)
        
        self.tfc_h = nn.Conv2d(2*dim, dim, (1, 7), stride=1, padding=(0, 7//2), groups=dim, bias=False)
        self.tfc_w = nn.Conv2d(2*dim, dim, (7, 1), stride=1, padding=(7//2, 0), groups=dim, bias=False)
        self.reweight = Mlp(dim, dim // 4, dim * 3)
        self.proj = nn.Conv2d(dim, dim, 1, 1, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)   
        self.mode = mode
        
        if mode == 'fc':
            self.theta_h_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=True), nn.BatchNorm2d(dim), nn.ReLU())
            self.theta_w_conv = nn.Sequential(nn.Conv2d(dim, dim, 1, 1, bias=True), nn.BatchNorm2d(dim), nn.ReLU())
        else:
            self.theta_h_conv = nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False), nn.BatchNorm2d(dim), nn.ReLU())
            self.theta_w_conv = nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False), nn.BatchNorm2d(dim), nn.ReLU())

    def forward(self, x):
     
        B, C, H, W = x.shape
        theta_h = self.theta_h_conv(x)
        theta_w = self.theta_w_conv(x)

        x_h = self.fc_h(x)
        x_w = self.fc_w(x)
        x_h = torch.cat([x_h*torch.cos(theta_h),x_h*torch.sin(theta_h)], dim=1)
        x_w = torch.cat([x_w*torch.cos(theta_w),x_w*torch.sin(theta_w)], dim=1)

#         x_1=self.fc_h(x)
#         x_2=self.fc_w(x)
#         x_h=torch.cat([x_1*torch.cos(theta_h),x_2*torch.sin(theta_h)],dim=1)
#         x_w=torch.cat([x_1*torch.cos(theta_w),x_2*torch.sin(theta_w)],dim=1)
        
        h = self.tfc_h(x_h)
        w = self.tfc_w(x_w)
        c = self.fc_c(x)
        a = F.adaptive_avg_pool2d(h + w + c, output_size=1)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        x = h * a[0] + w * a[1] + c * a[2]
        x = self.proj(x)
        x = self.proj_drop(x)           
        return x


class WaveBlock(nn.Module):

    def __init__(self, dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.BatchNorm2d, mode='fc'):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PATM(dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop,mode=mode)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) 
        x = x + self.drop_path(self.mlp(self.norm2(x))) 
        return x


class PatchEmbedOverlapping(nn.Module):
    def __init__(self, patch_size=16, stride=16, padding=0, in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm2d, groups=1,use_norm=True):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.patch_size = patch_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding, groups=groups)      
        self.norm = norm_layer(embed_dim) if use_norm == True else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_embed_dim, out_embed_dim, patch_size,norm_layer=nn.BatchNorm2d,use_norm=True):
        super().__init__()
        assert patch_size == 2, patch_size
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=1)
        self.norm = norm_layer(out_embed_dim) if use_norm == True else nn.Identity()

    def forward(self, x):
        x = self.proj(x) 
        x = self.norm(x)
        return x


def basic_blocks(dim, index, layers, mlp_ratio=3., qkv_bias=False, qk_scale=None, attn_drop=0.,
                 drop_path_rate=0., norm_layer=nn.BatchNorm2d, mode='fc', **kwargs):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(WaveBlock(dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                      attn_drop=attn_drop, drop_path=block_dpr, norm_layer=norm_layer, mode=mode))
    blocks = nn.Sequential(*blocks)
    return blocks


class WaveMLP(nn.Module):
    def __init__(self,
                 layers=[3, 4, 18],
                 embed_dims=[64, 128, 320],
                 transitions=[True, True, True],
                 mlp_ratios=[8, 8, 4],
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 mode='fc',
                 ds_use_norm=False,
                 affine=True, track_running_stats=True):

        super().__init__()

        # norm_layer = partial(MyNorm, affine=affine, track_running_stats=track_running_stats)
        norm_layer = MyNorm

        self.patch_embed = PatchEmbedOverlapping(patch_size=7, stride=4, padding=2, in_chans=3,
                                                 embed_dim=embed_dims[0],
                                                 norm_layer=norm_layer,
                                                 use_norm=ds_use_norm)

        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers, mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                                 qk_scale=qk_scale, attn_drop=attn_drop_rate, drop_path_rate=drop_path_rate,
                                 norm_layer=norm_layer, mode=mode)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if transitions[i] or embed_dims[i] != embed_dims[i+1]:
                patch_size = 2 if transitions[i] else 1
                network.append(Downsample(embed_dims[i], embed_dims[i+1], patch_size,
                                          norm_layer=norm_layer, use_norm=ds_use_norm))

        self.network = nn.ModuleList(network)

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        return x

    def forward_tokens(self, x):
        for idx, block in enumerate(self.network):
            x = block(x)
        return x

    def forward(self, x):
        x = self.forward_embeddings(x)
        x = self.forward_tokens(x)
        return x
