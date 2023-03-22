# DDPM

One Diffusion model implementation base on Pytorch, feel free to check the full same c++ implementation
[https://github.com/GlassyWing/TorchDiffusion](https://github.com/GlassyWing/TorchDiffusion) If you want faster training speed and lower memory usage.
(The weight file could be load at any side)

## Supported Sampler

- [x] DDPM
- [x] DDIM
- [x] RectifiedFlow

## Usage

### Train

```
python train.py --dataset <dataset_dir> -t ddim -s 4
```

The default will create one `experiments` folder used to save the checkpoints and log images.

see more with `python train.py -h`

### Inference

Please check that scripts:

1. ddim_test.py
2. ddpm_test.py
3. rectifiedflow_test.py

## Generate Example


After 5 days (140k images):

<img src="./assets/bs_1024_epoch_91.png">

It's trained with this strategy:



| Epochs  | Approximate Batch Size | Batch Size | accumulation_steps |
|---------|------------------------|------------|--------------------|
| 0-40    | 64                     | 64         | 1                  |
| 40-80   | 128                    | 64         | 2                  |
| 80-120  | 256                    | 64         | 4                  |
| 120-160 | 512                    | 64         | 8                  |
| 160-200 | 1024                   | 64         | 16                 |

With this cmd:

```shell
python train.py -p <path/to/last_checkpoint> -t 'ddim' -s 4 -b 32 --accum <accumulation_steps>
```

**Tips:** You can still accumulate the size of the Batch_size to get better results
## QA

1. How long does it take to train to see considerable results?

   About 30min on 3090.

2. Memory usage?

   Image Size: 128 x 128

   Batch Size: 32

   Memory: 3GB

   Vedio Memroy: 17GB
