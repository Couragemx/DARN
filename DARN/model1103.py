import torch
from torch import nn
import ToolClass as tool


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


##---------- Spatial Attention Block----------
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                               dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.RReLU() if relu else None

    def forward(self, x):
        x = self.conv1(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class spatial_attn_layer(nn.Module):
    def __init__(self, kernel_size=5):
        super(spatial_attn_layer, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, dilation=1, relu=False)

    def forward(self, x):
        # import pdb;pdb.set_trace()
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        # broadcasting
        return x + x * scale


##########################################################################
## ------ Channel Attention Block--------------
class ca_layer(nn.Module):
    def __init__(self, channel, reduction=8, bias=True):
        super(ca_layer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias,dilation=1),
            nn.RReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias,dilation=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x + x * y


##########################################################################
##---------- Dual Attention Residual Block (DARB) ----------
class DAU(nn.Module):
    def __init__(
            self, n_feat, kernel_size=3, reduction=8,
            bias=False, act=nn.RReLU()):
        super(DAU, self).__init__()
        modules_body = [conv(n_feat, n_feat, kernel_size, bias=bias), act, conv(n_feat, n_feat, kernel_size, bias=bias)]
        self.body = nn.Sequential(*modules_body)

        ## Spatial Attention
        self.SA = spatial_attn_layer()

        ## Channel Attention
        self.CA = ca_layer(n_feat, reduction, bias=bias)

        self.conv1x1 = nn.Conv2d(n_feat * 2, n_feat, kernel_size=1, bias=bias)

    def forward(self, x):
        res = self.body(x)
        sa_branch = self.SA(res)
        ca_branch = self.CA(res)
        res = torch.cat([sa_branch, ca_branch], dim=1)
        res = self.conv1x1(res)
        res += x
        return res


class DARCNNv8(nn.Module):
    def __init__(self, channels=1, num_of_layers=2):
        super(DARCNNv8, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size,
                      padding=padding, dilation=1, bias=False),
            nn.RReLU(inplace=True)
        )

        self.dau = nn.Sequential(
            DAU(n_feat=64),
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      dilation=1, bias=False),
            # nn.BatchNorm2d(features),
            # nn.RReLU(inplace=True),

            DAU(n_feat=64),
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      dilation=1, bias=False),
            nn.BatchNorm2d(features),
            nn.RReLU(inplace=True),

            DAU(n_feat=64),
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      dilation=1, bias=False),
            # nn.BatchNorm2d(features),
            # nn.RReLU(inplace=True),

            DAU(n_feat=64),
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      dilation=1, bias=False),
            # nn.BatchNorm2d(features),
            # nn.RReLU(inplace=True),

            DAU(n_feat=64),
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      dilation=1, bias=False),

            DAU(n_feat=64),
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      dilation=1, bias=False),
            nn.BatchNorm2d(features),
            nn.RReLU(inplace=True)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size,
                      padding=padding, bias=False)
        )

    def forward(self, x):
        conv_1 = self.conv_1(x)
        dau = self.dau(conv_1)
        res = self.conv_2(dau)
        res += x
        return res


if __name__ == '__main__':
    model = DARCNNv8()
    tool.print_network(model)
