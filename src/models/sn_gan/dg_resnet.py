import torch.nn as nn
import torch.nn.functional as F

from .snconv2d import SNConv2d
from .snlinear import SNLinear


class ResBlockDiscriminator(nn.Module):
    def __init__(
        self, in_channels, out_channels, hidden_channels=None, use_BN=False, downsample=False
    ):
        super(ResBlockDiscriminator, self).__init__()
        # self.conv1 = SNConv2d(n_dim, n_out, kernel_size=3, stride=2)
        hidden_channels = in_channels
        self.downsample = downsample

        self.resblock = self.make_res_block(
            in_channels, out_channels, hidden_channels, use_BN, downsample
        )
        self.residual_connect = self.make_residual_connect(in_channels, out_channels)

    def make_res_block(self, in_channels, out_channels, hidden_channels, use_BN, downsample):
        model = []
        if use_BN:
            model += [nn.BatchNorm2d(in_channels)]

        model += [nn.ReLU()]
        model += [SNConv2d(in_channels, hidden_channels, kernel_size=3, padding=1)]
        model += [nn.ReLU()]
        model += [SNConv2d(hidden_channels, out_channels, kernel_size=3, padding=1)]
        if downsample:
            model += [nn.AvgPool2d(2)]
        return nn.Sequential(*model)

    def make_residual_connect(self, in_channels, out_channels):
        model = []
        model += [SNConv2d(in_channels, out_channels, kernel_size=1, padding=0)]
        if self.downsample:
            model += [nn.AvgPool2d(2)]
            return nn.Sequential(*model)
        else:
            return nn.Sequential(*model)

    def forward(self, input):
        return self.resblock(input) + self.residual_connect(input)


class OptimizedBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OptimizedBlock, self).__init__()
        self.res_block = self.make_res_block(in_channels, out_channels)
        self.residual_connect = self.make_residual_connect(in_channels, out_channels)

    def make_res_block(self, in_channels, out_channels):
        model = []
        model += [SNConv2d(in_channels, out_channels, kernel_size=3, padding=1)]
        model += [nn.ReLU()]
        model += [SNConv2d(out_channels, out_channels, kernel_size=3, padding=1)]
        model += [nn.AvgPool2d(2)]
        return nn.Sequential(*model)

    def make_residual_connect(self, in_channels, out_channels):
        model = []
        model += [SNConv2d(in_channels, out_channels, kernel_size=1, padding=0)]
        model += [nn.AvgPool2d(2)]
        return nn.Sequential(*model)

    def forward(self, input):
        return self.res_block(input) + self.residual_connect(input)


class Discriminator(nn.Module):
    def __init__(self, ndf=64, ndlayers=4):
        super(Discriminator, self).__init__()
        self.res_d = self.make_model(ndf, ndlayers)
        self.fc = nn.Sequential(SNLinear(ndf * 16, 1), nn.Sigmoid())

    def make_model(self, ndf, ndlayers):
        model = []
        model += [OptimizedBlock(3, ndf)]
        tndf = ndf
        for i in range(ndlayers):
            model += [ResBlockDiscriminator(tndf, tndf * 2, downsample=True)]
            tndf *= 2
        model += [nn.ReLU()]
        return nn.Sequential(*model)

    def forward(self, input):
        out = self.res_d(input)
        out = F.avg_pool2d(out, out.size(3), stride=1)

        out = out.view(-1, 1024)
        return self.fc(out)


class ResBlockGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=None, upsample=False):
        super(ResBlockGenerator, self).__init__()
        # self.conv1 = SNConv2d(n_dim, n_out, kernel_size=3, stride=2)
        hidden_channels = in_channels
        self.upsample = upsample
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.conv_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.upsampling = nn.UpsamplingBilinear2d(scale_factor=2)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(hidden_channels)
        self.relu = nn.ReLU()

    def forward_residual_connect(self, input):
        out = self.conv_sc(input)
        if self.upsample:
            out = self.upsampling(out)
        # out = self.upconv2(out)
        return out

    def forward(self, input):
        out = self.relu(self.bn1(input))
        out = self.conv1(out)
        if self.upsample:
            out = self.upsampling(out)
            # out = self.upconv1(out)
        out = self.relu(self.bn2(out))
        out = self.conv2(out)
        out_res = self.forward_residual_connect(input)
        return out + out_res


class Generator(nn.Module):
    def __init__(self, ngf, z=128, nlayers=4):
        super(Generator, self).__init__()
        self.input_layer = nn.Linear(z, (4**2) * ngf * 16)
        self.generator = self.make_model(ngf, nlayers)

    def make_model(self, ngf, nlayers):
        model = []
        tngf = ngf * 16
        for _ in range(nlayers):
            model += [ResBlockGenerator(tngf, tngf / 2, upsample=True)]
            tngf /= 2
        model += [nn.BatchNorm2d(ngf)]
        model += [nn.ReLU()]
        model += [nn.Conv2d(ngf, 3, kernel_size=3, stride=1, padding=1)]
        model += [nn.Tanh()]
        return nn.Sequential(*model)

    def forward(self, z):
        out = self.input_layer(z)
        out = out.view(z.size(0), -1, 4, 4)
        out = self.generator(out)

        return out
