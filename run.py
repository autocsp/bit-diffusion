from bit_diffusion import Unet, Transformer, Trainer, BitDiffusion
import torch
from torch import nn




# model = Unet(
#     dim = 32,
#     channels = 1,
#     dim_mults = (1, 3),
# ).cuda()

model = Transformer(
    dim = 32,
    channels = 1
).cuda()

bit_diffusion = BitDiffusion(
    model,
    image_size = 10,
    timesteps = 100,
    time_difference = 0.1,       # they found in the paper that at lower number of timesteps, a time difference during sampling of greater than 0 helps FID. as timesteps increases, this time difference can be set to 0 as it does not help
    use_ddim = True              # use ddim
).cuda()

trainer = Trainer(
    bit_diffusion,
    'sudoku.npy',             # path to your folder of images
    results_folder = './results',     # where to save results
    num_samples = 16,                 # number of samples
    train_batch_size = 16,             # training batch size
    gradient_accumulate_every = 4,    # gradient accumulation
    train_lr = 1e-4,                  # learning rate
    save_and_sample_every = 5000,     # how often to save and sample
    train_num_steps = 250000,          # total training steps
    ema_decay = 0.995,                # exponential moving average decay
)

trainer.train()
