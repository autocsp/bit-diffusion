from bit_diffusion import Unet, Trainer, BitDiffusion
import torch
from torch import nn

class Block(nn.Module):

    def __init__(self, embed_dim=8, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        # self.mlp = nn.ModuleDict(dict(
        #     c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
        #     c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
        #     act     = NewGELU(),
        #     dropout = nn.Dropout(config.resid_pdrop),
        # ))
        # m = self.mlp
        # self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x, time, x_self_cond = None):
        orig_shape = x.shape
        x = x.reshape((orig_shape[0], -1, self.embed_dim))
        inp = self.norm1(x)
        x = x + self.attn(inp, inp, inp, need_weights=False)[0]
        # x = x + self.mlpf(self.ln_2(x))
        return x.reshape(orig_shape)

    @property
    def channels(self):
        return self.embed_dim


model = Unet(
    dim = 32,
    channels = 1,
    dim_mults = (1, 3),
)#.cuda()

# model = Block(embed_dim=4, num_heads=4)

bit_diffusion = BitDiffusion(
    model,
    image_size = 10,
    timesteps = 100,
    time_difference = 0.1,       # they found in the paper that at lower number of timesteps, a time difference during sampling of greater than 0 helps FID. as timesteps increases, this time difference can be set to 0 as it does not help
    use_ddim = True              # use ddim
)#.cuda()

trainer = Trainer(
    bit_diffusion,
    'sudoku.npy',             # path to your folder of images
    results_folder = './results',     # where to save results
    num_samples = 16,                 # number of samples
    train_batch_size = 8,             # training batch size
    gradient_accumulate_every = 4,    # gradient accumulation
    train_lr = 1e-4,                  # learning rate
    save_and_sample_every = 1000,     # how often to save and sample
    train_num_steps = 20000,          # total training steps
    ema_decay = 0.995,                # exponential moving average decay
)

trainer.train()
