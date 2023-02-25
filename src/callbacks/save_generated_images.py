"""Callback to Save Generated Images from GAN"""
import os

import pytorch_lightning as pl
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
            save_image(
                gen_imgs[j],
                os.path.join(output_dir, f"{counter}.png"),
            )
            counter += 1


class SaveGeneratedImages(pl.Callback):
    """Saves Generated Images: For Training and Testing during MIA Attack
    A Callback that during the training process saves generated images `every_k_epochs` and
    in the last epoch

    Parameters
    ----------
    output_dir: str
        Output directory with respect to where the generated images are stored
    every_k_epochs: int
        Images will be stored every k epochs
    number_of_images: int
        Number of images will be stored
    batch_size: int
        Batch size of generation
    """

    def __init__(
        self,
        output_dir: str,
        every_k_epochs: int = 50,
        number_of_images: int = 1000,
        batch_size: int = 100,
    ):
        super().__init__()
        self.output_dir = output_dir
        self.every_k_epochs = every_k_epochs
        self.number_of_images = number_of_images
        self.batch_size = batch_size
        self.gen_images_dirs = []
        self.gen_test_images_dirs = []

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        current_epoch = pl_module.current_epoch
        condition = (
            (current_epoch % self.every_k_epochs == 0)
            or (current_epoch == trainer.max_epochs - 1)
            and (current_epoch > 0)
        )
        if condition:
            # generate images for training
            gen_images_dir = os.path.join(self.output_dir, f"gen_images_{current_epoch}/")
            os.makedirs(gen_images_dir, exist_ok=True)
            generate_and_save_images(
                gan=pl_module,
                number_of_images=self.number_of_images,
                output_dir=os.path.join(gen_images_dir, "0"),
                batch_size=self.batch_size,
            )
            self.gen_images_dirs.append(gen_images_dir)

            # generate images for testing
            gen_test_images_dir = os.path.join(self.output_dir, f"gen_test_images_{current_epoch}/")
            os.makedirs(gen_test_images_dir, exist_ok=True)
            generate_and_save_images(
                gan=pl_module,
                number_of_images=self.number_of_images,
                output_dir=os.path.join(gen_test_images_dir, "0"),
                batch_size=self.batch_size,
            )
            self.gen_test_images_dirs.append(gen_test_images_dir)
