import torch

from ddpm.models import Unet, DDIM, DDPM,RectifiedFlow

if __name__ == '__main__':
    img_size = 128
    scales = [1, 1, 2, 2, 4, 4]
    emb_dim = 64
    T = 250
    stride = 4
    attn_resolution = 16

    torch.manual_seed(0)
    model = Unet(3, img_size, scales, emb_dim=emb_dim, attn_resolution=attn_resolution)
    model.load_state_dict(torch.load("RectifiedFlow_ckpt_1_0.pth", map_location="cpu"))
    # print(model.state_dict())
    # model.load_state_dict(torch.load("experiments/checkpoints/sample_ckpt_0_5000.pth", map_location="cpu"))
    diffusion = RectifiedFlow(model, img_size, T, emb_dim, stride=stride)
    diffusion.to("cuda:0")
    diffusion = diffusion.eval()

    diffusion.sample(device="cuda:0", path=f"./demo_2_{T}.png")