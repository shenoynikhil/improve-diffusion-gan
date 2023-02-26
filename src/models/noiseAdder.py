import torch
import os.path as osp
import os
from .utils import sample_image

class BaseAdder(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def set_gan(self, gan):
        self.gan = gan
    
    @torch.no_grad()
    def apply_noise(self, t_img, gan):
        raise NotImplementedError()

    @torch.no_grad()
    def log_noise(self, pl_module):
        return


class IdentiyAdder(BaseAdder):
    """
    returns the identity
    """
    def apply_noise(self, t_img, gan):
        return t_img
    
    

class GNadder(BaseAdder):
    """
    Gan noise adder
    add noise by: label_img = noise_w*GAN_out + (1-noise_w)*true_img
    """
    def __init__(self, noise_w=0.2, cache_size=1000, gan=None):
        super().__init__()
        self.gan = None
        self.noise_cache = None      # gan noise cache
        self.cache_size = cache_size
        self.noise_w = noise_w       # noise weight
        self.n_runned = 0            # Gnadder run count 
        self.n_update_noise = 20   # interval to update noise cache by
        self.generated_noise = None

    
    @torch.no_grad()
    def update_noise_cache(self, gan):
        self.noise_cache = gan.generate_images(self.cache_size)
    
    def inc_count(self):
        if self.n_runned >= self.n_update_noise:
            self.n_runned = 1

        self.n_runned += 1


    @torch.no_grad()
    def apply_noise(self, t_img:torch.Tensor, gan = None):
        """
        t_img (torch.Tensor): true image
        """
        if self.noise_cache is None or (self.n_runned%self.n_runned == 0):
            self.update_noise_cache(gan)
    
        sample_idxs = torch.randint(self.cache_size, (t_img.shape[0],))
        self.inc_count()
        self.generated_noise = self.noise_w*self.noise_cache[sample_idxs].to(t_img.device) + (1-self.noise_w)*t_img
        return self.generated_noise
    

    @torch.no_grad()
    def log_noise(self, pl_module):
        save_path = osp.join(pl_module.output_dir, "noise_img")
        os.makedirs(save_path, exist_ok=True)
        if pl_module.current_epoch%5 == 0:
            noise = self.generated_noise[:100]
            sample_image(
                gen_imgs=noise,
                n_row=10,
                epochs_done=pl_module.current_epoch,
                output_dir=save_path
            )            