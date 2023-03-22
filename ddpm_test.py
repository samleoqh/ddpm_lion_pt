from ddpm.models import Unet, DDPM

if __name__ == '__main__':
    img_size = 128
    scales = [1, 1, 2, 2, 4, 4]
    emb_dim = 64
    T = 1000

    model = Unet(3, img_size, scales, emb_dim=emb_dim)
    diffusion = DDPM(model, img_size, T, emb_dim)

    diffusion.sample(device="cpu")