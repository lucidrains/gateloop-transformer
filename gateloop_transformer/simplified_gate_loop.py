from functools import partial

import torch
from torch import nn, Tensor
from torch.nn import Module

from typing import Tuple

from einops import rearrange, pack, unpack
from einops.layers.torch import Rearrange

from gateloop_transformer.gateloop_transformer import RMSNorm
from gateloop_transformer.associative_scan import associative_scan

# plain pytorch non-fused associative scan

def exists(v):
    return v is not None

def abs_clamp_eps(t, eps = 1e-20):
    sign = torch.sign(t)
    return sign * t.abs().clamp(min = eps)

# associative scan using heinsen sequences
# https://github.com/glassroom/heinsen_sequence
# graciously shared to the world by Franz A. Heinsen in https://arxiv.org/abs/2311.06281 in October 2023

def heinsen_associative_scan(a, kv, eps = 1e-20):
    log_a = a.clamp(min = eps).log()
    log_kv = abs_clamp_eps(kv, eps = eps).to(dtype = torch.complex64).log()

    a_star = torch.cumsum(log_a, dim = 1)
    log_x0_plus_b_star = torch.logcumsumexp(log_kv - a_star, dim = 1)
    log_x = a_star + log_x0_plus_b_star
    return a_star.exp().real, log_x.exp().real

# naive associative scan with some torchscript of binary operator

@torch.jit.script
def binary_operator(
    a: Tuple[Tensor, Tensor],
    b: Tuple[Tensor, Tensor]
):
    a_i, kv_i = a
    a_j, kv_j = b
    return a_j * a_i, torch.addcmul(kv_j, a_j, kv_i)

# gate loop operator

def gate_loop_operator(q, kv, a, cache = None, heinsen = False):

    if exists(cache):
        cache_a, cache_kv = cache
        a, a_ps = pack([cache_a, a], 'b * d')
        kv, kv_ps = pack([cache_kv, kv], 'b * d')

    if heinsen:
        a, kv = heinsen_associative_scan(a, kv)
    else:
        a, kv = associative_scan(binary_operator, (a, kv))

    if exists(cache):
        _, a = unpack(a, a_ps, 'b * d')
        _, kv = unpack(kv, kv_ps, 'b * d')

    return q * kv, (a[:, -1], kv[:, -1])

# using jax associative scan

def get_jax_gate_loop_operator():
    try:
        from jax import jit, numpy as jnp
        from jax.lax import associative_scan
        from jax2torch import jax2torch
    except ImportError as e:
        print(f'jax and jax2torch must be installed - `pip install jax2torch`')

    @jit
    def jax_gate_loop_operator(q, kv, a, cache = None):
        def binary_operator(e_i, e_j):
            a_i, kv_i = e_i
            a_j, kv_j = e_j
            return a_j * a_i, a_j * kv_i + kv_j

        if exists(cache):
            cache_a, cache_kv = cache
            a, a_ps = pack([cache_a, a], 'b * d')
            kv, kv_ps = pack([cache_kv, kv], 'b * d')

        _, y = associative_scan(binary_operator, (a, kv), axis = 1)

        if exists(cache):
            _, a = unpack(a, a_ps, 'b * d')
            _, kv = unpack(kv, kv_ps, 'b * d')

        return q * y, (a[:, -1], kv[:, -1])

    return jax2torch(jax_gate_loop_operator)

# simple gate loop layer

class SimpleGateLoopLayer(Module):
    """
    simplified gate loop
    seeing if it can supplement attention as shown in https://github.com/lucidrains/mega-pytorch
    """

    def __init__(
        self,
        dim,
        prenorm = True,
        use_heinsen = False,
        use_jax_associative_scan = False,
        post_ln = False,
        reverse = False
    ):
        super().__init__()
        assert (int(use_heinsen) + int(use_jax_associative_scan)) <= 1

        self.norm = RMSNorm(dim) if prenorm else nn.Identity()

        self.dim = dim

        self.to_qkva = nn.Sequential(
            nn.Linear(dim, dim * 3, bias = False),
            Rearrange('b n (qkva d) -> qkva (b d) n 1', qkva = 3)
        )

        self.use_heinsen = use_heinsen
        self.use_jax = use_jax_associative_scan

        if use_jax_associative_scan:
            self.gate_loop_fn = get_jax_gate_loop_operator()
        elif use_heinsen:
            self.gate_loop_fn = partial(gate_loop_operator, heinsen = True)
        else:
            self.gate_loop_fn = gate_loop_operator

        self.maybe_post_ln = nn.LayerNorm(dim) if post_ln else nn.Identity()
        self.split_heads = Rearrange('(b d) n 1 -> b n d', d = dim)

        self.reverse = reverse

    def forward(
        self,
        x,
        cache = None,
        return_cache = False
    ):
        if self.reverse:
            x = torch.flip(x, dims = (-2,))

        x = self.norm(x)

        q, kv, a = self.to_qkva(x)

        out, cache = self.gate_loop_fn(q, kv, a.sigmoid(), cache = cache)

        out = self.split_heads(out)
        out = self.maybe_post_ln(out)

        if self.reverse:
            out = torch.flip(out, dims = (-2,))

        if not return_cache:
            return out

        assert not self.reverse, 'caching only works with non-reversed seq'

        return out, cache
