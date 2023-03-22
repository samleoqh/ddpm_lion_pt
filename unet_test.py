import torch

from ddpm.models.unet import Unet

if __name__ == '__main__':
    with torch.no_grad():
        model = Unet(3, 64, [1, 1, 2, 2, 4, 4], emb_dim=64)
        inp = torch.randn(1, 3, 64, 64)
        t = torch.randn(1, 64)
        out = model(inp, t)
        print(out.shape)