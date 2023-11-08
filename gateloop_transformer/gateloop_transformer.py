import torch
from torch.nn import Module, ModuleList
from torch import nn, einsum, Tensor
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

# helpers

def exists(v):
    return v is not None

def safe_cumprod(t, eps = 1e-10, dim = -1):
    t = torch.clip(t, min = eps, max = 1.)
    return torch.exp(torch.cumsum(torch.log(t), dim = dim))

# rms norm

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

# feedforward

def FeedForward(dim, mult = 4):
    dim_inner = dim * mult
    return nn.Sequential(
        RMSNorm(dim),
        nn.Linear(dim, dim_inner),
        nn.GELU(),
        nn.Linear(dim_inner, dim)
    )

# attention

class CausalFullAttention(Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        data_dependent_rel_pos = False,
        frac_gradient_data_dependent_rel_pos = 0.5
    ):
        super().__init__()
        dim_inner = dim_head * heads

        self.scale = dim_head ** -0.5

        self.norm = RMSNorm(dim)

        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3),
            Rearrange('b n (qkv h d) -> qkv b h n d', h = heads, qkv = 3)
        )

        self.data_dependent_rel_pos = data_dependent_rel_pos
        self.frac_gradient_data_dependent_rel_pos = frac_gradient_data_dependent_rel_pos

        if data_dependent_rel_pos:
            self.to_a = nn.Sequential(
                nn.Linear(dim, dim_inner),
                Rearrange('b n (h d) -> b h n d', h = heads)
            )

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(dim_inner, dim)
        )

    def forward(self, x):
        x = self.norm(x)

        q, k, v = self.to_qkv(x)

        q = q * self.scale

        if self.data_dependent_rel_pos:
            frac_gradient = self.frac_gradient_data_dependent_rel_pos

            a = self.to_a(x)

            # allow for data dependent relative position projection to change more slowly
            # alternative to using a lowered learning rate mentioned in paper

            a = a * frac_gradient + a.detach() * (1 - frac_gradient)

            a = a.sigmoid() # not sure about this

            a_cumprod = safe_cumprod(a, dim = -2)
            a_cumprod_inverse = 1. / a_cumprod.clamp(min = 1e-10)

            q = q * a_cumprod
            k = k * a_cumprod_inverse

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        if not self.data_dependent_rel_pos:
            attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        return self.to_out(out)

# main class

class Transformer(Module):
    def __init__(
        self,
        dim,
        *,
        num_tokens,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        data_dependent_rel_pos = False,
        frac_gradient_data_dependent_rel_pos = 0.5
    ):
        super().__init__()

        self.token_emb = nn.Embedding(num_tokens, dim)

        layers = ModuleList([])
        for _ in range(depth):
            layers.append(ModuleList([
                CausalFullAttention(
                    dim = dim,
                    dim_head = dim_head,
                    heads = heads,
                    data_dependent_rel_pos = data_dependent_rel_pos,
                    frac_gradient_data_dependent_rel_pos = frac_gradient_data_dependent_rel_pos
                ),
                FeedForward(
                    dim = dim,
                    mult = ff_mult
                )
            ]))

        self.layers = ModuleList(layers)

        self.to_logits = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, num_tokens, bias = False)
        )

    def forward(self, x):
        x = self.token_emb(x)

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        logits = self.to_logits(x)

        return logits
