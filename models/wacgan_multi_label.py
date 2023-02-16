"""LightningModule to setup WACGAN_GP_MultiLabel setup.
"""
import torch
import torch.nn.functional as F

from .wacgan import WACGAN_GP, Generator


class GeneratorMultiLabel(Generator):
    def forward(self, z, label):
        x = torch.cat((z, label.unsqueeze(2).unsqueeze(3)), dim=1)
        x = self.main_module(x)
        return self.output(x)


class WACGAN_GP_MultiLabel(WACGAN_GP):
    """WACGAN_GP_MultiLabel Network
    An Extension of WACGAN_GP with MultiLabel Logit Sigmoid Loss
    """

    def __init__(self, opt):
        super().__init__(opt)
        self.generator = GeneratorMultiLabel(opt)

    def _generate_labels(self, size):
        return torch.randint(2, (size, self.opt.n_classes)).float()

    def auxiliary_loss(self, y_hat, y):
        """F.binary_cross_entropy_with_logits between y_hat and y"""
        return F.binary_cross_entropy_with_logits(y_hat, y, reduction="mean")

    def generate_images(self, batch_size: int):
        """Generate Images function"""
        with torch.no_grad():
            return self(
                torch.randn((batch_size, self.opt.latent_dim, 1, 1)).to(self.device),
                torch.randint(2, (batch_size, self.opt.n_classes)).to(self.device),
            )
