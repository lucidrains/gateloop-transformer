from torch import nn
from torch.nn import Module

from einops import rearrange
from einops.layers.torch import Rearrange

from gateloop_transformer.gateloop_transformer import RMSNorm
from gateloop_transformer.associative_scan import associative_scan

def gate_loop_operator(q, kv, a):
    def binary_operator(a, b):
        a_i, kv_i = a
        a_j, kv_j = b
        return a_j * a_i, a_j * kv_i + kv_j

    _, kv = associative_scan(binary_operator, (a, kv))

    return q * kv

class SimpleGateLoopLayer(Module):
    """
    simplified gate loop
    seeing if it can supplement attention as shown in https://github.com/lucidrains/mega-pytorch
    """

    def __init__(self, dim):
        super().__init__()
        self.norm = RMSNorm(dim)

        self.dim = dim

        self.to_qkva = nn.Sequential(
            nn.Linear(dim, dim * 3, bias = False),
            Rearrange('b n (qkva d) -> qkva (b d) n 1', qkva = 3)
        )

        self.split_heads = Rearrange('(b d) n 1 -> b n d', d = dim)

    def forward(self, x):
        x = self.norm(x)

        q, kv, a = self.to_qkva(x)

        out = gate_loop_operator(q, kv, a.sigmoid())

        return self.split_heads(out)
