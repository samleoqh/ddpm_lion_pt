import torch
from torchvision.utils import save_image
from tqdm import tqdm

from ddpm.models import DDPM
import numpy as np

__all__ = ["DDIM"]


class DDIM(DDPM):

    def __init__(self, model, img_size, T=1000, embedding_size=2, stride=2, eta=1):
        super().__init__(model, img_size, T, embedding_size)

        self.stride = stride
        self.eta = eta

        self.register_buffer("bar_alpha_", self.bar_alpha[::self.stride])
        self.register_buffer("bar_alpha_pre_", torch.from_numpy(np.pad(self.bar_alpha_.cpu().numpy()[:-1], [1, 0], constant_values=1)))
        self.register_buffer("bar_beta_", torch.sqrt(1 - self.bar_alpha_ ** 2))
        self.register_buffer("bar_beta_pre_", torch.sqrt(1 - self.bar_alpha_pre_ ** 2))
        self.register_buffer("alpha_", self.bar_alpha_ / self.bar_alpha_pre_)
        self.register_buffer("sigma_", self.bar_beta_pre_ / self.bar_beta_ * torch.sqrt(1 - self.alpha_ ** 2) * self.eta)
        self.register_buffer("epsilon_", self.bar_beta_ - self.alpha_ * torch.sqrt(self.bar_beta_pre_ ** 2 - self.sigma_ ** 2))

    @torch.no_grad()
    def sample(self, path=None, n=4, z_samples=None, t0=0, device="cuda"):

        T_ = len(self.bar_alpha_)
        if z_samples is None:
            z_samples = torch.randn(n ** 2, 3, self.img_size, self.img_size, device=device)
        else:
            z_samples = z_samples.copy()
        for t in tqdm(range(t0, T_), ncols=0):
            t = T_ - t - 1
            bt = torch.tensor([t * self.stride] * z_samples.shape[0], dtype=torch.long, device=device)
            z_samples = z_samples - self.epsilon_[t] * self.forward(z_samples, bt)
            z_samples = z_samples / self.alpha_[t]
            z_samples = z_samples + torch.randn_like(z_samples) * self.sigma_[t]
        x_samples = torch.clip(z_samples, -1, 1)
        if path is None:
            return x_samples
        save_image(x_samples, path, nrow=n, normalize=True)
