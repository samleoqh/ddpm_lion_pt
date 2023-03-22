from copy import deepcopy

import torch
import os

from tqdm import tqdm

from .optim.lion import Lion
from .utils import update_average


class Trainer:

    def __init__(self, sampler,
                 exp_name="",
                 train_batch_size=32,
                 train_lr=2e-4,
                 train_num_epochs=10000,
                 ema_decay=0.995,
                 num_workers=2,
                 save_and_sample_every=100,
                 accumulation_steps=2,
                 ):
        self.sampler = sampler
        self.train_batch_size = train_batch_size
        self.train_lr = train_lr / accumulation_steps
        self.train_num_epochs = train_num_epochs
        self.ema_decay = ema_decay
        self.num_workers = num_workers
        self.save_and_sample_every = save_and_sample_every
        self.accumulation_steps = accumulation_steps

        self.sample_path = os.path.join("experiments", exp_name, "outputs")
        self.checkpoint_path = os.path.join("experiments", exp_name, "checkpoints")
        os.makedirs(self.checkpoint_path, exist_ok=True)
        os.makedirs(self.sample_path, exist_ok=True)

        self.sampler_shadow = deepcopy(sampler)
        self.ema_updater = update_average
        update_average(self.sampler_shadow, sampler, beta=0.)

        self.device = next(self.sampler.parameters()).device

    def train(self, dataloader):
        train_dataloader = dataloader

        optimizer = Lion(self.sampler.parameters(), lr=self.train_lr)

        step = 0
        for epoch in range(self.train_num_epochs):

            self.sampler.zero_grad(set_to_none=True)

            with tqdm(train_dataloader) as train_bar:
                for idx, x_real in enumerate(train_bar):
                    x_real = x_real.to(self.device)

                    # with torch.cuda.amp.autocast():
                    noise_images, steps, noise = self.sampler.p_x(x_real)
                    denoise = self.sampler(noise_images, steps)
                    loss = torch.sum((denoise - noise) ** 2, dim=[1, 2, 3], keepdim=True).mean()

                    # scaler.scale(loss).backward()
                    loss.backward()

                    if (step + 1) % self.accumulation_steps == 0:
                        # scaler.step(optimizer)
                        # scaler.update()
                        optimizer.step()
                        self.sampler.zero_grad(set_to_none=True)

                        self.ema_updater(self.sampler_shadow, self.sampler, beta=self.ema_decay)
                        # scheduler.step()

                    train_bar.set_description(f"[{epoch}/{self.train_num_epochs}] "
                                              f"loss: {loss.item():.4f} "
                                              f"lr: {optimizer.state_dict()['param_groups'][0]['lr']:.6f}")
                    step += 1

                    if step % self.save_and_sample_every == 0:
                        self.sampler.sample(f"{self.sample_path}/sample_ckpt_{epoch}_{step}.png",
                                                   4, device=self.device)
                        self.sampler_shadow.sample(f"{self.sample_path}/sample_ckpt_{epoch}_{step}_ema.png",
                                                   4, device=self.device)
                        torch.save(self.sampler_shadow.model.state_dict(),
                                   f"{self.checkpoint_path}/sample_ckpt_{epoch}_{step}.pth")


