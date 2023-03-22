import argparse
import logging

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from ddpm.datasets import ImageFolderDataset
from ddpm.models import DDPM, Unet, DDIM
from ddpm.models.rectified_flow import RectifiedFlow
from ddpm.trainer import Trainer
from ddpm.utils import weights_init

torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    parser = argparse.ArgumentParser(description="Trainer for Diffusion model.")
    parser.add_argument("--dataset_path", "-d", type=str, required=True, help="dataset path")
    parser.add_argument("--epochs", type=int, default=10000, help="number of epochs.")
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="size of each sample batch")
    parser.add_argument("--pretrained_weights", "-p", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--name", type=str, default="", help="experiment name")
    parser.add_argument("--type", "-t", type=str, default="rflow", help="Sampler type [ddpm/ddim/rflow], Default, `rflow`")
    parser.add_argument("--stride", "-s", type=int, default=1, help="sample stride for ddim")
    parser.add_argument("--num_steps", "-n", type=int, default=1000, help="sample times. Default, 1000")
    parser.add_argument("--accum", type=int, default=1, help="accumulation steps, Default, 1.")

    opt = parser.parse_args()
    img_size = 128
    scales = [1, 1, 2, 2, 4, 4]
    emb_dim = 64
    attn_resolution = 16
    T = opt.num_steps
    sampler = opt.type
    stride = opt.stride
    accum = opt.accum

    tfms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])

    ds = ImageFolderDataset(opt.dataset_path, img_size, transform=tfms)
    dataloader = DataLoader(ds, batch_size=opt.batch_size, num_workers=opt.n_cpu,
                            persistent_workers=True,
                            )

    model = Unet(3, img_size, scales, emb_dim=emb_dim, attn_resolution=attn_resolution)
    if opt.pretrained_weights is not None:
        pretrained_dict = torch.load(opt.pretrained_weights, map_location="cpu")
        model_dict = model.state_dict()

        # Fiter out unneccessary keys
        filtered_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        print("load pretrained weights!")

    if sampler == "ddpm":
        diffusion = DDPM(model, img_size, T, emb_dim)
    elif sampler == "ddim":
        diffusion = DDIM(model, img_size, T, emb_dim, stride=stride)  # Sample steps = T // stride. E.g, 1000 / 50 = 20
    elif sampler == "rflow":
        diffusion = RectifiedFlow(model, img_size, T, emb_dim)
    else:
        raise ValueError("Unsupported sampler type")

    if opt.pretrained_weights is None:
        diffusion = diffusion.apply(weights_init())

    diffusion.cuda(0)

    trainer = Trainer(
        diffusion,
        train_batch_size=opt.batch_size,
        train_lr=2e-4,
        train_num_epochs=opt.epochs,
        ema_decay=0.995,
        save_and_sample_every=500,
        num_workers=opt.n_cpu,
        accumulation_steps=accum,
    )
    trainer.train(dataloader)
