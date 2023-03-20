from torch import nn

from .spectral_norm import SpectralNorm


class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100, channels: int = 3):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim

        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, channels, 3, stride=1, padding=(1, 1)),
            nn.Tanh(),
        )

    def forward(self, z):
        return self.model(z.view(-1, self.latent_dim, 1, 1))


class Discriminator(nn.Module):
    def __init__(self, channels: int = 3, leak: float = 0.1, w_g: int = 4):
        super(Discriminator, self).__init__()
        self.leak = leak
        self.w_g = w_g

        self.conv1 = SpectralNorm(nn.Conv2d(channels, 64, 3, stride=1, padding=(1, 1)))

        self.conv2 = SpectralNorm(nn.Conv2d(64, 64, 4, stride=2, padding=(1, 1)))
        self.conv3 = SpectralNorm(nn.Conv2d(64, 128, 3, stride=1, padding=(1, 1)))
        self.conv4 = SpectralNorm(nn.Conv2d(128, 128, 4, stride=2, padding=(1, 1)))
        self.conv5 = SpectralNorm(nn.Conv2d(128, 256, 3, stride=1, padding=(1, 1)))
        self.conv6 = SpectralNorm(nn.Conv2d(256, 256, 4, stride=2, padding=(1, 1)))
        self.conv7 = SpectralNorm(nn.Conv2d(256, 512, 3, stride=1, padding=(1, 1)))

        self.fc = SpectralNorm(nn.Linear(self.w_g * self.w_g * 512, 1))

    def forward(self, x):
        m = x
        m = nn.LeakyReLU(self.leak)(self.conv1(m))
        m = nn.LeakyReLU(self.leak)(self.conv2(m))
        m = nn.LeakyReLU(self.leak)(self.conv3(m))
        m = nn.LeakyReLU(self.leak)(self.conv4(m))
        m = nn.LeakyReLU(self.leak)(self.conv5(m))
        m = nn.LeakyReLU(self.leak)(self.conv6(m))
        m = nn.LeakyReLU(self.leak)(self.conv7(m))

        return self.fc(m.view(-1, self.w_g * self.w_g * 512))
