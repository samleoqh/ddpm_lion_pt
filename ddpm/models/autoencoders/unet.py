from collections import OrderedDict
from torch import nn

__all__ = ["Unet"]

from ..layers import *


class Unet(nn.Module):

    def __init__(self, img_c,
                 img_size,
                 scales,
                 emb_dim,
                 min_pixel=4,
                 attn_resolution=32,
                 n_block=2,
                 n_groups=32,
                 out_c=3,
                 with_time_emb=True):
        super().__init__()

        self.n_block = 2
        self.img_c = img_c
        self.scales = scales
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.n_groups = n_groups
        self.out_c = out_c
        self.with_time_emb = with_time_emb
        self.attn_resolution = attn_resolution

        self.stem = nn.Conv2d(img_c, emb_dim, (3, 3), padding=1)

        min_img_size = min(self.img_size)
        skip_pooling = 0
        cur_c = emb_dim
        encoder_blocks = OrderedDict()
        chs = []
        for i, scale in enumerate(scales):
            for j in range(n_block):
                chs.append([cur_c, scale * emb_dim])
                block = ResidualBlock(cur_c, scale * emb_dim, emb_dim, n_groups, self.with_time_emb)
                cur_c = scale * emb_dim
                encoder_blocks[f'enc_block_{i * n_block + j}'] = block

            if min_img_size <= self.attn_resolution:
                encoder_blocks[f'attn_enc_block_{i * n_block}'] = AttentionBlock(cur_c, 8, cur_c // 8)

            if min_img_size > min_pixel:
                encoder_blocks[f'down_block_{i}'] = Downsample(cur_c)
                min_img_size = min_img_size // 2
            else:
                skip_pooling += 1
        chs_inv = chs[::-1]
        encoder_blocks[f'enc_block_{(len(scales)) * n_block}'] = ResidualBlock(cur_c, cur_c, emb_dim, n_groups,
                                                                               self.with_time_emb)
        self.encoder_blocks = nn.ModuleDict(encoder_blocks)
        decoder_blocks = OrderedDict()
        for i, scale in enumerate(scales[::-1]):
            if i >= skip_pooling:
                decoder_blocks[f'up_block_{i}'] = Upsample(cur_c)
                min_img_size *= 2
            for j in range(n_block):
                decoder_blocks[f'dec_block_{i * n_block + j}'] = ResidualBlock(*chs_inv[i * n_block + j][::-1],
                                                                               emb_dim,
                                                                               n_groups, self.with_time_emb)
                cur_c = chs_inv[i * n_block + j][0]

            if min_img_size <= self.attn_resolution:
                decoder_blocks[f'attn_dec_block_{i * n_block}'] = AttentionBlock(cur_c, 8, cur_c // 8)
        decoder_blocks[f'to_rgb'] = nn.Sequential(GroupNormCustom(n_groups, cur_c),
                                                  nn.SiLU(inplace=True),
                                                  nn.Conv2d(cur_c, out_c, (3, 3), padding=1, bias=False))
        self.decoder_blocks = nn.ModuleDict(decoder_blocks)

    def forward(self, x, t=None):
        x = self.stem(x)

        inners = [x]
        for name, module in self.encoder_blocks.items():
            if name.startswith("enc"):
                x = module(x, t)
                inners.append(x)
            elif name.startswith("attn"):
                x = module(x)
            else:
                x = module(x)
                inners.append(x)

        inners = inners[:-2]
        for name, module in self.decoder_blocks.items():

            if name.startswith("up"):
                x = module(x)
                xi = inners.pop()
                x = x + xi

            elif name.startswith("dec"):
                xi = inners.pop()
                x = module(x, t)
                x = x + xi
            elif name.startswith("attn"):
                x = module(x)
            else:
                x = module(x)
        return x
