"""Utils"""
import os

import torch
import torch.nn.functional as F
from torchvision.utils import save_image


def generate_and_save_images(
    gan,
    number_of_images: int,
    output_dir: str,
    batch_size: int = 100,
):
    """Generates number_of_images using generator of the GAN"""
    gan.eval()
    os.makedirs(output_dir, exist_ok=True)
    counter = 0
    epochs = int(number_of_images / batch_size)
    for _ in range(epochs):
        gen_imgs = gan.generate_images(batch_size=batch_size)
        # denormalize images, as we normalize images
        gen_imgs = 0.5 + (gen_imgs * 0.5)
        for j in range(gen_imgs.shape[0]):
            save_image(gen_imgs[j], os.path.join(output_dir, f"{counter}.png"))
            counter += 1


def focal_binary_cross_entropy(inputs, targets, gamma=2, alpha=1, logits=True, reduce=True):
    """Computes focal_loss given inputs and targets"""
    if logits:
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
    else:
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
    pt = torch.exp(-BCE_loss)
    F_loss = alpha * (1 - pt) ** gamma * BCE_loss

    if reduce:
        return torch.mean(F_loss)
    else:
        return F_loss
