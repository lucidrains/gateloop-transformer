import torch
from torch import nn, Tensor
from torch.nn import Module

from typing import Tuple

from einops import rearrange
from einops.layers.torch import Rearrange

from gateloop_transformer.gateloop_transformer import RMSNorm
from gateloop_transformer.associative_scan import associative_scan

# plain pytorch non-fused associative scan

def gate_loop_operator(q, kv, a):

    @torch.jit.script
    def binary_operator(
        a: Tuple[Tensor, Tensor],
        b: Tuple[Tensor, Tensor]
    ):
        a_i, kv_i = a
        a_j, kv_j = b
        return a_j * a_i, torch.addcmul(kv_j, a_j, kv_i)

    _, kv = associative_scan(binary_operator, (a, kv))

    return q * kv

# using jax associative scan

def get_jax_gate_loop_operator():
    try:
        from jax import jit, numpy as jnp
        from jax.lax import associative_scan
        from jax2torch import jax2torch
    except ImportError as e:
        print(f'jax and jax2torch must be installed - `pip install jax2torch`')

    @jit
    def jax_gate_loop_operator(q, kv, a):
        def binary_operator(e_i, e_j):
            a_i, kv_i = e_i
            a_j, kv_j = e_j
            return a_j * a_i, a_j * kv_i + kv_j

        _, y = associative_scan(binary_operator, (a, kv), axis = 1)

        return q * y

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
        use_jax_associative_scan = False,
        reverse = False
    ):
        super().__init__()
        self.norm = RMSNorm(dim) if prenorm else nn.Identity()

        self.dim = dim

        self.to_qkva = nn.Sequential(
            nn.Linear(dim, dim * 3, bias = False),
            Rearrange('b n (qkva d) -> qkva (b d) n 1', qkva = 3)
        )

        if use_jax_associative_scan:
            self.gate_loop_fn = get_jax_gate_loop_operator()
        else:
            self.gate_loop_fn = gate_loop_operator

        self.split_heads = Rearrange('(b d) n 1 -> b n d', d = dim)

        self.reverse = reverse

    def forward(self, x):

        if self.reverse:
            x = torch.flip(x, dims = (-2,))

        x = self.norm(x)

        q, kv, a = self.to_qkva(x)

        out = self.gate_loop_fn(q, kv, a.sigmoid())

        out = self.split_heads(out)

        if self.reverse:
            out = torch.flip(out, dims = (-2,))

        return out
