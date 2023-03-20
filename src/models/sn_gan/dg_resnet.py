# ResNet generator and discriminator
import numpy as np
from torch import nn

from .spectral_norm import SpectralNorm

channels = 3


class ResBlockGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.0)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.0)

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2,
        )
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Upsample(scale_factor=2)

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class ResBlockDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.0)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.0)

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(), SpectralNorm(self.conv1), nn.ReLU(), SpectralNorm(self.conv2)
            )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                SpectralNorm(self.conv1),
                nn.ReLU(),
                SpectralNorm(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0),
            )
        self.bypass = nn.Sequential()
        if stride != 1:

            self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

            self.bypass = nn.Sequential(
                SpectralNorm(self.bypass_conv), nn.AvgPool2d(2, stride=stride, padding=0)
            )
            # if in_channels == out_channels:
            #     self.bypass = nn.AvgPool2d(2, stride=stride, padding=0)
            # else:
            #     self.bypass = nn.Sequential(
            #         SpectralNorm(nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)),
            #         nn.AvgPool2d(2, stride=stride, padding=0)
            #     )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.0)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.0)
        nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            SpectralNorm(self.conv1), nn.ReLU(), SpectralNorm(self.conv2), nn.AvgPool2d(2)
        )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            SpectralNorm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class Generator(nn.Module):
    def __init__(self, z_dim, channels: int = 3, gen_size: int = 128):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.gen_size = gen_size

        self.dense = nn.Linear(self.z_dim, 4 * 4 * gen_size)
        self.final = nn.Conv2d(gen_size, channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform(self.dense.weight.data, 1.0)
        nn.init.xavier_uniform(self.final.weight.data, 1.0)

        self.model = nn.Sequential(
            ResBlockGenerator(gen_size, gen_size, stride=2),
            ResBlockGenerator(gen_size, gen_size, stride=2),
            ResBlockGenerator(gen_size, gen_size, stride=2),
            nn.BatchNorm2d(gen_size),
            nn.ReLU(),
            self.final,
            nn.Tanh(),
        )

    def forward(self, z):
        return self.model(self.dense(z).view(-1, self.gen_size, 4, 4))


class Discriminator(nn.Module):
    def __init__(self, disc_size: int = 128):
        super(Discriminator, self).__init__()
        self.disc_size = disc_size

        self.model = nn.Sequential(
            FirstResBlockDiscriminator(channels, disc_size, stride=2),
            ResBlockDiscriminator(disc_size, disc_size, stride=2),
            ResBlockDiscriminator(disc_size, disc_size),
            ResBlockDiscriminator(disc_size, disc_size),
            nn.ReLU(),
            nn.AvgPool2d(8),
        )
        self.fc = nn.Linear(disc_size, 1)
        nn.init.xavier_uniform(self.fc.weight.data, 1.0)
        self.fc = SpectralNorm(self.fc)

    def forward(self, x):
        return self.fc(self.model(x).view(-1, self.disc_size))
