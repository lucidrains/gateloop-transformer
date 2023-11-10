from typing import List, Tuple, Callable

import numpy as onp
from jax import random, jit, nn, lax, numpy as np
from jax.lax import associative_scan

from equinox import Module, static_field

from einops import rearrange, repeat

# linear

class Linear(Module):
    weight: np.ndarray
    bias: np.ndarray

    def __init__(self, dim_in, dim_out, *, key):
        weight_key, bias_key = random.split(key)
        self.weight = random.normal(weight_key, (dim_in, dim_out))
        self.bias = random.normal(bias_key, (dim_out,))

    def __call__(self, x, *, key = None):
        return x @ self.weight + self.bias

# rmsnorm

class RMSNorm(Module):
    scale: float = static_field()
    eps: float = static_field()
    gamma: np.ndarray

    def __init__(self, dim, eps = 1e-5):
        self.eps = eps
        self.scale = dim ** 0.5
        self.gamma = np.ones((dim,))

    def __call__(self, x):
        sum_of_squares = np.sum(np.square(x), axis = -1, keepdims = True)
        inv_norm = lax.rsqrt(sum_of_squares + self.eps)
        return inv_norm * x * self.gamma * self.scale

# gate loop layer

class GateLoop(Module):
    norm: RMSNorm

    def __init__(
        self,
        dim,
        key
    ):
        self.norm = RMSNorm(dim)

    def __call__(self, x):
        return self.norm(x)

# feedforward

class FeedForward(Module):
    norm: RMSNorm
    proj_in: Linear
    proj_out: Linear

    def __init__(
        self,
        *,
        dim,
        key,
        mult = 4,
    ):
        self.norm = RMSNorm(dim)
        self.proj_in = Linear(dim, dim * mult, key = key)
        self.proj_out = Linear(dim * mult, dim, key = key)

    def __call__(self, x):
        x = self.norm(x)
        x = self.proj_in(x)
        x = nn.gelu(x)
        x = self.proj_out(x)
        return x

# main class

class GateLoopTransformer(Module):
    embedding: np.ndarray
    norm: Module
    layers: List[Tuple[GateLoop, FeedForward]]

    def __init__(
        self,
        *,
        num_tokens,
        dim,
        dim_head,
        depth,
        heads,
        key,
        ff_mult = 4
    ):
        self.embedding = random.normal(key, (num_tokens, dim)) * 0.02

        layers = []

        for _ in range(depth):
            gateloop = GateLoop(dim = dim, key = key)

            ff = FeedForward(dim = dim, mult = ff_mult, key = key)

            layers.append((gateloop, ff))

        self.layers = layers

        self.norm = RMSNorm(dim)

    @jit
    def __call__(self, x):
        n = x.shape[-1]
        x = self.embedding[x]

        for gateloop, ff in self.layers:
            x = gateloop(x) + x
            x = ff(x) + x

        x = self.norm(x)
        return x @ self.embedding.transpose()
