import torch
import torch.nn as nn

from ..utils import weights_init_normal
from .snconv2d import SNConv2d
from .snlinear import SNLinear


class ConditionalGenerator(nn.Module):
    """Conditional Generator
    Generates images given a latent vector and a label
    """

    def __init__(
        self,
        latent_dim: int,
        channels: int,
        img_size: int,
        n_classes: int,
    ):
        super(ConditionalGenerator, self).__init__()
        assert img_size == 32 or img_size == 28, "Final size must be 32 or 28"
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.channels = channels

        # create label embedding
        self.label_emb = nn.Embedding(n_classes, latent_dim)

        self.init_size = img_size // 4  # Initial size before upsampling
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size**2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

        # intialize with normal weights
        self.apply(weights_init_normal)

    def forward(self, noise, labels):
        gen_input = torch.mul(self.label_emb(labels), noise)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class ConditionalDiscriminator(nn.Module):
    """Conditional Discriminator
    Critic that takes an image and a label as input
    and provides a real or fake image prediction (2 classes)
    and a label prediction (n_classes)
    """

    def __init__(self, channels: int, img_size: int, n_classes: int):
        super(ConditionalDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [
                SNConv2d(in_filters, out_filters, 3, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout2d(0.25),
            ]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2**4

        # Output layers
        self.adv_layer = nn.Sequential(SNLinear(128 * ds_size**2, 1))
        self.aux_layer = nn.Sequential(SNLinear(128 * ds_size**2, n_classes))

        # intialize with normal weights
        self.apply(weights_init_normal)

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        # output real/fake labels (validity) and the digit labels (0-9)
        return validity, label
