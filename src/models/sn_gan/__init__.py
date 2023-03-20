from .dg import Discriminator, Generator
from .dg_resnet import Discriminator as ResnetDiscriminator
from .dg_resnet import Generator as ResnetGenerator
from .model import SpectralNormGAN

__all__ = [
    "Generator",
    "Discriminator",
    "ResnetGenerator",
    "ResnetDiscriminator",
    "SpectralNormGAN",
]
