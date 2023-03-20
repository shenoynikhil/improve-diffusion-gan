from typing import List

import torch.nn as nn

from .snconv2d import SNConv2d


class Generator(nn.Module):
    """Generator Framework for WGAN-GP"""

    def __init__(self, latent_dim: int, channels: int, final_size: int = 32):
        super().__init__()
        # Filters [1024, 512, 256]
        # Input_dim = 100
        # Output_dim = C (number of channels)
        assert final_size == 32 or final_size == 28, "Final size must be 32 or 28"
        self.latent_dim = latent_dim
        self.channels = channels
        self.final_size = final_size

        self.main_module = nn.Sequential(
            # Z latent vector 100
            # (1 - 1) * 1 + 1 * (4 - 1) + 1 = 4 -> (b, 1024, 4, 4) for final_size = 32/28
            nn.ConvTranspose2d(
                in_channels=latent_dim,
                out_channels=1024,
                kernel_size=4,
                stride=1,
                padding=0,
            ),
            nn.BatchNorm2d(num_features=1024),
            nn.ReLU(True),
            # State (1024x4x4)
            # (4 - 1) * 2 - 2 * 1 + 1 * (4 - 1) + 1 = 8 -> (b, 512, 8, 8) for final_size = 32
            # (4 - 1) * 2 - 2 * 1 + 1 * (3 - 1) + 1 = 7 -> (b, 512, 7, 7) for final_size = 28
            nn.ConvTranspose2d(
                in_channels=1024,
                out_channels=512,
                kernel_size=4 if final_size == 32 else 3,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),
            # State (512x8x8)
            # (8 - 1) * 2 - 2 * 1 + 1 * (4 - 1) + 1 = 16 -> (b, 256, 16, 16) for final_size = 32
            # (7 - 1) * 2 - 2 * 1 + 1 * (4 - 1) + 1 = 14 -> (b, 256, 14, 14) for final_size = 28
            nn.ConvTranspose2d(
                in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),
            # State (256x16x16)
            # (16 - 1) * 2 - 2 * 1 + 1 * (4 - 1) + 1 = 32 -> (b, 128, 32, 32)
            # (14 - 1) * 2 - 2 * 1 + 1 * (4 - 1) + 1 = 28 -> (b, 128, 28, 28)
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=channels,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
        )
        # output of main module --> Image (Cx32x32)

        self.output = nn.Tanh()

    def forward(self, z):
        return self.output(self.main_module(z))


class Discriminator(nn.Module):
    """Discriminator for WACGAN-GP"""

    def __init__(self, channels: int, conv_channel_list: List[int] = [128, 256, 512]):
        """Initialize the Discriminator

        Parameters
        ----------
        channels : int
            Number of channels in the input image
        conv_channel_list : List[int], optional, default=[128, 256, 512]
            List of input_channel, output_channel for each convolutional layer.
            if len(conv_channel_list) = 3, then the discriminator will have
            3 convolutional layers.
            Ensure len(conv_channel_list) < 5 (kernel size = 4
            can only have 4 conv layers)
        """
        super().__init__()
        assert len(conv_channel_list) < 5, "With kernel size = 4, max 4 conv layers"
        channel_list = [channels] + conv_channel_list
        conv_list = []
        for i in range(len(channel_list) - 1):
            conv_list.extend(
                [
                    SNConv2d(
                        in_channels=channel_list[i],
                        out_channels=channel_list[i + 1],
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                ]
            )
        self.main_module = nn.Sequential(
            # Omitting batch normalization in critic because our new penalized training
            # objective (WGAN with gradient penalty) is no longer valid
            # in this setting, since we penalize the norm of the critic's gradient with
            # respect to each input independently and not the enitre batch.
            # There is not good & fast implementation of layer normalization -->
            # using per instance normalization nn.InstanceNorm2d() Image (Cx32x32)
            *conv_list,
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        # output of main module --> State (1024x4x4)
        self.fc = nn.Sequential(nn.Linear(channel_list[-1], 128), nn.LeakyReLU(0.2, inplace=True))

        self.output = nn.Linear(128, 1)

    def forward(self, x):
        x = self.main_module(x)
        x = x.view(x.size()[0], -1).flatten(1)
        x = self.fc(x)
        return self.output(x)

    def feature_extraction(self, x):
        # Use discriminator for feature extraction then flatten to vector of 16384
        x = self.main_module(x)
        return x.view(-1, 1024 * 4 * 4)
