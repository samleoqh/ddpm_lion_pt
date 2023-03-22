import torch
from torch import nn
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm

__all__ = ["DDPM"]

from .pos_embs.sinusoidal import position_encoding_1d


class DDPM(nn.Module):

    def __init__(self, model, img_size, T=1000, embedding_size=2):
        super().__init__()
        self.img_size = img_size
        self.T = T

        self.register_buffer("alpha", torch.sqrt(1 - 0.02 * torch.arange(1, T + 1) / T))
        self.register_buffer("beta", torch.sqrt(1 - torch.pow(self.alpha, 2)))
        self.register_buffer("bar_alpha", torch.cumprod(self.alpha, 0))
        self.register_buffer("bar_beta", torch.sqrt(1 - self.bar_alpha.pow(2)))
        self.register_buffer("sigma", self.beta.clone())
        self.register_buffer("t", position_encoding_1d(embedding_size, T))
        self.model = model

    def p_x(self, x_real):
        batch_images = x_real
        batch_size = len(batch_images)
        batch_steps = np.random.choice(self.T, batch_size, replace=False)
        batch_steps = torch.from_numpy(batch_steps).long().to(x_real.device)

        batch_bar_alpha = self.bar_alpha[batch_steps].float()
        batch_bar_beta = self.bar_beta[batch_steps].float()

        batch_bar_alpha = batch_bar_alpha.reshape(-1, 1, 1, 1)
        batch_bar_beta = batch_bar_beta.reshape(-1, 1, 1, 1)

        batch_noise = torch.randn_like(batch_images)
        batch_noise_images = batch_images * batch_bar_alpha + batch_noise * batch_bar_beta
        return batch_noise_images, batch_steps, batch_noise

    def forward(self, x_in, t_in):
        t_e = self.t[t_in]  # (n, z)
        x_r = self.model(x_in, t_e)
        return x_r

    @torch.no_grad()
    def sample(self, path=None, n=4, z_samples=None, t0=0, device="cuda"):
        if z_samples is None:
            z_samples = torch.randn(n ** 2, 3, self.img_size, self.img_size, device=device)
        else:
            z_samples = z_samples.copy()
        for t in tqdm(range(t0, self.T), ncols=0):
            t = self.T - t - 1
            bt = torch.tensor([t] * z_samples.shape[0], dtype=torch.long, device=device)
            z_samples = z_samples - self.beta[t] ** 2 / self.bar_beta[t] * self.forward(z_samples, bt)
            z_samples = z_samples / self.alpha[t]
            z_samples = z_samples + torch.randn_like(z_samples) * self.sigma[t]
        x_samples = torch.clip(z_samples, -1, 1)
        if path is None:
            return x_samples
        save_image(x_samples, path, nrow=n, normalize=True)
