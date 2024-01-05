import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.search.modules import WaveMLP_Strip, ResNetBlock, ViTBlock
from lib.models.activation import MishAuto as Mish


OPS = {
    # WaveMLPBlock
    'wavemlp_k3_t4': lambda C_in, C_out, stride, affine, track_running_stats: WaveMLP(C_in, C_out, 3, 1, t=4, affine=affine, track_running_stats=track_running_stats),
    'wavemlp_k3_t8': lambda C_in, C_out, stride, affine, track_running_stats: WaveMLP(C_in, C_out, 3, 1, t=8, affine=affine, track_running_stats=track_running_stats),
    'wavemlp_k5_t4': lambda C_in, C_out, stride, affine, track_running_stats: WaveMLP(C_in, C_out, 5, 2, t=4, affine=affine, track_running_stats=track_running_stats),
    'wavemlp_k5_t8': lambda C_in, C_out, stride, affine, track_running_stats: WaveMLP(C_in, C_out, 5, 2, t=8, affine=affine, track_running_stats=track_running_stats),
    'wavemlp_k7_t4': lambda C_in, C_out, stride, affine, track_running_stats: WaveMLP(C_in, C_out, 7, 3, t=4, affine=affine, track_running_stats=track_running_stats),
    'wavemlp_k7_t8': lambda C_in, C_out, stride, affine, track_running_stats: WaveMLP(C_in, C_out, 7, 3, t=8, affine=affine, track_running_stats=track_running_stats),
    'wavemlp_k3_t4_dw': lambda C_in, C_out, stride, affine, track_running_stats: WaveMLP(C_in, C_out, 3, 1, t=4, dw=True, affine=affine, track_running_stats=track_running_stats),
    'wavemlp_k3_t8_dw': lambda C_in, C_out, stride, affine, track_running_stats: WaveMLP(C_in, C_out, 3, 1, t=8, dw=True, affine=affine, track_running_stats=track_running_stats),
    'wavemlp_k5_t4_dw': lambda C_in, C_out, stride, affine, track_running_stats: WaveMLP(C_in, C_out, 5, 2, t=4, dw=True, affine=affine, track_running_stats=track_running_stats),
    'wavemlp_k5_t8_dw': lambda C_in, C_out, stride, affine, track_running_stats: WaveMLP(C_in, C_out, 5, 2, t=8, dw=True, affine=affine, track_running_stats=track_running_stats),
    'wavemlp_k7_t4_dw': lambda C_in, C_out, stride, affine, track_running_stats: WaveMLP(C_in, C_out, 7, 3, t=4, dw=True, affine=affine, track_running_stats=track_running_stats),
    'wavemlp_k7_t8_dw': lambda C_in, C_out, stride, affine, track_running_stats: WaveMLP(C_in, C_out, 7, 3, t=8, dw=True, affine=affine, track_running_stats=track_running_stats),
    # WaveMLP_Strip_Block: the block limits constant size, not catering to tracking problems
    'wavemlp_strip_t1_31': lambda C_in, C_out, stride, affine, track_running_stats: WaveMLP_Strip(C_in, C_out, t=1, HW=31, affine=affine, track_running_stats=track_running_stats),
    'wavemlp_strip_t3_31': lambda C_in, C_out, stride, affine, track_running_stats: WaveMLP_Strip(C_in, C_out, t=3, HW=31, affine=affine, track_running_stats=track_running_stats),
    'wavemlp_strip_t6_31': lambda C_in, C_out, stride, affine, track_running_stats: WaveMLP_Strip(C_in, C_out, t=6, HW=31, affine=affine, track_running_stats=track_running_stats),
    'wavemlp_strip_t1_29': lambda C_in, C_out, stride, affine, track_running_stats: WaveMLP_Strip(C_in, C_out, t=1, HW=29, affine=affine, track_running_stats=track_running_stats),
    'wavemlp_strip_t3_29': lambda C_in, C_out, stride, affine, track_running_stats: WaveMLP_Strip(C_in, C_out, t=3, HW=29, affine=affine, track_running_stats=track_running_stats),
    'wavemlp_strip_t6_29': lambda C_in, C_out, stride, affine, track_running_stats: WaveMLP_Strip(C_in, C_out, t=6, HW=29, affine=affine, track_running_stats=track_running_stats),
    'wavemlp_strip_t1_16': lambda C_in, C_out, stride, affine, track_running_stats: WaveMLP_Strip(C_in, C_out, t=1, HW=16, affine=affine, track_running_stats=track_running_stats),
    'wavemlp_strip_t3_16': lambda C_in, C_out, stride, affine, track_running_stats: WaveMLP_Strip(C_in, C_out, t=3, HW=16, affine=affine, track_running_stats=track_running_stats),
    'wavemlp_strip_t6_16': lambda C_in, C_out, stride, affine, track_running_stats: WaveMLP_Strip(C_in, C_out, t=6, HW=16, affine=affine, track_running_stats=track_running_stats),
    # ResNetBlock: for ablation of post-fusion
    'resnet_k3_t4': lambda C_in, C_out, stride, affine, track_running_stats: ResNetBlock(C_in, C_out, 3, 1, t=4, affine=affine, track_running_stats=track_running_stats),
    'resnet_k3_t8': lambda C_in, C_out, stride, affine, track_running_stats: ResNetBlock(C_in, C_out, 3, 1, t=8, affine=affine, track_running_stats=track_running_stats),
    'resnet_k5_t4': lambda C_in, C_out, stride, affine, track_running_stats: ResNetBlock(C_in, C_out, 5, 2, t=4, affine=affine, track_running_stats=track_running_stats),
    'resnet_k5_t8': lambda C_in, C_out, stride, affine, track_running_stats: ResNetBlock(C_in, C_out, 5, 2, t=8, affine=affine, track_running_stats=track_running_stats),
    'resnet_k7_t4': lambda C_in, C_out, stride, affine, track_running_stats: ResNetBlock(C_in, C_out, 7, 3, t=4, affine=affine, track_running_stats=track_running_stats),
    'resnet_k7_t8': lambda C_in, C_out, stride, affine, track_running_stats: ResNetBlock(C_in, C_out, 7, 3, t=8, affine=affine, track_running_stats=track_running_stats),
    'resnet_k3_t4_dw': lambda C_in, C_out, stride, affine, track_running_stats: ResNetBlock(C_in, C_out, 3, 1, t=4, dw=True, affine=affine, track_running_stats=track_running_stats),
    'resnet_k3_t8_dw': lambda C_in, C_out, stride, affine, track_running_stats: ResNetBlock(C_in, C_out, 3, 1, t=8, dw=True, affine=affine, track_running_stats=track_running_stats),
    'resnet_k5_t4_dw': lambda C_in, C_out, stride, affine, track_running_stats: ResNetBlock(C_in, C_out, 5, 2, t=4, dw=True, affine=affine, track_running_stats=track_running_stats),
    'resnet_k5_t8_dw': lambda C_in, C_out, stride, affine, track_running_stats: ResNetBlock(C_in, C_out, 5, 2, t=8, dw=True, affine=affine, track_running_stats=track_running_stats),
    'resnet_k7_t4_dw': lambda C_in, C_out, stride, affine, track_running_stats: ResNetBlock(C_in, C_out, 7, 3, t=4, dw=True, affine=affine, track_running_stats=track_running_stats),
    'resnet_k7_t8_dw': lambda C_in, C_out, stride, affine, track_running_stats: ResNetBlock(C_in, C_out, 7, 3, t=8, dw=True, affine=affine, track_running_stats=track_running_stats),
    # ViTBlock: for ablation of post-fusion, vit blocks have no difference about kernel size, padding, dw, affine and track_running_stats
    'vit_k3_t4': lambda C_in, C_out, stride, affine, track_running_stats: ViTBlock(C_in, C_out, t=4),
    'vit_k3_t8': lambda C_in, C_out, stride, affine, track_running_stats: ViTBlock(C_in, C_out, t=8),
    'vit_k5_t4': lambda C_in, C_out, stride, affine, track_running_stats: ViTBlock(C_in, C_out, t=4),
    'vit_k5_t8': lambda C_in, C_out, stride, affine, track_running_stats: ViTBlock(C_in, C_out, t=8),
    'vit_k7_t4': lambda C_in, C_out, stride, affine, track_running_stats: ViTBlock(C_in, C_out, t=4),
    'vit_k7_t8': lambda C_in, C_out, stride, affine, track_running_stats: ViTBlock(C_in, C_out, t=8),
    'vit_k3_t4_dw': lambda C_in, C_out, stride, affine, track_running_stats: ViTBlock(C_in, C_out, t=4),
    'vit_k3_t8_dw': lambda C_in, C_out, stride, affine, track_running_stats: ViTBlock(C_in, C_out, t=8),
    'vit_k5_t4_dw': lambda C_in, C_out, stride, affine, track_running_stats: ViTBlock(C_in, C_out, t=4),
    'vit_k5_t8_dw': lambda C_in, C_out, stride, affine, track_running_stats: ViTBlock(C_in, C_out, t=8),
    'vit_k7_t4_dw': lambda C_in, C_out, stride, affine, track_running_stats: ViTBlock(C_in, C_out, t=4),
    'vit_k7_t8_dw': lambda C_in, C_out, stride, affine, track_running_stats: ViTBlock(C_in, C_out, t=8),
    # Skip the operation
    'skip_connect': lambda C_in, C_out, stride, affine, track_running_stats: Skip(C_in, C_out, 1, affine=affine, track_running_stats=track_running_stats),
}


class WaveMLP(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, padding, t=3, dw=False, affine=True, track_running_stats=True):
        super(WaveMLP, self).__init__()

        self._proj = nn.Conv2d(C_in, C_out, kernel_size=1, stride=1, padding=0)
        self._norm1 = nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats)
        self._norm2 = nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats)
        self._patm = PATM(C_out, kernel_size, padding, dw=dw, affine=affine, track_running_stats=track_running_stats)
        self._mlp = Mlp(in_features=C_out, hidden_features=C_out * t)

    def forward(self, x):
        x = self._proj(x)
        x = x + self._patm(self._norm1(x))
        x = x + self._mlp(self._norm2(x))
        return x


class PATM(nn.Module):
    def __init__(self, dim, kernel_size, padding, dw=False, affine=True, track_running_stats=True):
        super().__init__()

        self.fc_h = nn.Conv2d(dim, dim, 1, 1, bias=False)
        self.fc_w = nn.Conv2d(dim, dim, 1, 1, bias=False)
        self.fc_c = nn.Conv2d(dim, dim, 1, 1, bias=False)
        self.tfc_h = nn.Conv2d(2 * dim, dim, (1, kernel_size), stride=1, padding=(0, padding), groups=dim, bias=False)
        self.tfc_w = nn.Conv2d(2 * dim, dim, (kernel_size, 1), stride=1, padding=(padding, 0), groups=dim, bias=False)
        self.reweight = Mlp(dim, dim // 4, dim * 3)
        self.proj = nn.Conv2d(dim, dim, 1, 1, bias=True)
        if dw:
            self.theta_h_conv = nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),
                                              nn.BatchNorm2d(dim, affine=affine, track_running_stats=track_running_stats),
                                              nn.ReLU())
            self.theta_w_conv = nn.Sequential(nn.Conv2d(dim, dim, 3, stride=1, padding=1, groups=dim, bias=False),
                                              nn.BatchNorm2d(dim, affine=affine, track_running_stats=track_running_stats),
                                              nn.ReLU())
        else:
            self.theta_h_conv = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=True),
                                              nn.BatchNorm2d(dim, affine=affine, track_running_stats=track_running_stats),
                                              nn.ReLU())
            self.theta_w_conv = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0, bias=True),
                                              nn.BatchNorm2d(dim, affine=affine, track_running_stats=track_running_stats),
                                              nn.ReLU())

    def forward(self, x):

        B, C, H, W = x.shape
        theta_h = self.theta_h_conv(x)
        theta_w = self.theta_w_conv(x)

        x_h = self.fc_h(x)
        x_w = self.fc_w(x)
        x_h = torch.cat([x_h * torch.cos(theta_h), x_h * torch.sin(theta_h)], dim=1)
        x_w = torch.cat([x_w * torch.cos(theta_w), x_w * torch.sin(theta_w)], dim=1)

        h = self.tfc_h(x_h)
        w = self.tfc_w(x_w)
        c = self.fc_c(x)
        a = F.adaptive_avg_pool2d(h + w + c, output_size=1)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).contiguous()
        a = a.softmax(dim=0).unsqueeze(-1).unsqueeze(-1)
        x = h * a[0] + w * a[1] + c * a[2]
        x = self.proj(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.act = nn.GELU()
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Skip(nn.Module):
    def __init__(self, C_in, C_out, stride, affine=True, track_running_stats=True):
        super(Skip, self).__init__()
        if C_in != C_out:
            skip_conv = nn.Sequential(
                nn.Conv2d(C_in, C_out, kernel_size=1, stride=stride, padding=0, groups=1, bias=False),
                nn.BatchNorm2d(C_out, affine=affine, track_running_stats=track_running_stats))
            stride = 1
        self.op = Identity(stride)

        if C_in != C_out:
            self.op = nn.Sequential(skip_conv, self.op)

    def forward(self,x):
        return self.op(x)


class Identity(nn.Module):
    def __init__(self, stride):
        super(Identity, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x
        else:
            return x[:, :, ::self.stride, ::self.stride]
