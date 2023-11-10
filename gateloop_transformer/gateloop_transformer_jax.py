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

def gate_loop_operator(kv, q, a):

    def binary_operator(e_i, e_j):
        a_i, kv_i = e_i
        a_j, kv_j = e_j
        return a_j * a_i, a_j * kv_i + kv_j

    _, y = associative_scan(binary_operator, (a, kv), axis = 1)
    return q * y

class GateLoop(Module):
    norm: RMSNorm
    wq: np.ndarray
    wk: np.ndarray
    wv: np.ndarray
    wa: np.ndarray
    wg: np.ndarray
    wo: np.ndarray

    def __init__(
        self,
        dim,
        key
    ):
        """
        q - query
        k - key
        v - value
        a - state transition
        g - gating with silu activation
        o - output
        """

        q_key, k_key, v_key, a_key, g_key, o_key = random.split(key, 6)

        self.wq = random.normal(q_key, (dim, dim))
        self.wk = random.normal(k_key, (dim, dim))
        self.wv = random.normal(v_key, (dim, dim))
        self.wa = random.normal(a_key, (dim, dim))
        self.wg = random.normal(g_key, (dim, dim))
        self.wo = random.normal(o_key, (dim, dim))

        self.norm = RMSNorm(dim)

    def __call__(self, x):
        x = self.norm(x)

        q = x @ self.wq
        k = x @ self.wk
        v = x @ self.wv
        a = x @ self.wa
        g = x @ self.wg
        o = x @ self.wo

        kv = k * v
        a = nn.sigmoid(a)

        y = gate_loop_operator(kv, q, a)

        y = y * nn.silu(g)

        o = y @ self.wo

        return o

# basic feedforward with pre-rmsnorm

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
        x = self.embedding[x]

        for gateloop, ff in self.layers:
            x = gateloop(x) + x
            x = ff(x) + x

        x = self.norm(x)
        logits = x @ self.embedding.transpose()

        return logits
