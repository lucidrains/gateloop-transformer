from functools import partial

import torch
from torch.nn import Module, ModuleList
from torch import nn, einsum, Tensor
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F

from einops import rearrange
from einops.layers.torch import Rearrange

from rotary_embedding_torch import RotaryEmbedding

from gateloop_transformer.associative_scan import associative_scan

# helpers

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def Sequential(*modules):
    modules = list(filter(exists, modules))
    num_modules = len(modules)

    if num_modules == 0:
        return nn.Identity()
    elif num_modules == 1:
        return modules[0]

    return nn.Sequential(*modules)

# rms norm

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.scale * self.gamma

# norm wrappers

class PreNorm(Module):
    def __init__(self, dim, fn: Module):
        super().__init__()
        self.fn = fn
        self.norm = RMSNorm(dim)

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs) + x

class PostNorm(Module):
    def __init__(self, dim, fn: Module):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs) + x)

# feedforward

def FeedForward(dim, mult = 4):
    dim_inner = dim * mult
    return nn.Sequential(
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
        rotary_emb = False,
        add_swish_gating = False,
        data_dependent_rel_pos = False,
        frac_gradient_data_dependent_rel_pos = 0.5,
        softmax_normalize = None
    ):
        super().__init__()
        dim_inner = dim_head * heads
        self.softmax_normalize = default(softmax_normalize, not data_dependent_rel_pos)

        self.scale = dim_head ** -0.5

        self.rotary_emb = RotaryEmbedding(dim_head) if rotary_emb else None

        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias = False),
            Rearrange('b n (qkv h d) -> qkv b h n d', h = heads, qkv = 3)
        )

        self.data_dependent_rel_pos = data_dependent_rel_pos
        self.frac_gradient_data_dependent_rel_pos = frac_gradient_data_dependent_rel_pos

        if data_dependent_rel_pos:
            self.to_a = nn.Sequential(
                nn.Linear(dim, dim_inner, bias = False),
                Rearrange('b n (h d c) -> b h n d c', h = heads, c = 2)
            )

        self.to_gates = None

        if add_swish_gating:
            self.to_gates = nn.Sequential(
                nn.Linear(dim, dim_inner, bias = False),
                nn.SiLU(),
                Rearrange('b n (h d) -> b h n d', h = heads)
            )

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(dim_inner, dim)
        )

    def forward(
        self,
        x,
        ablate_complex = False,
        ablate_state_transition = False
    ):
        q, k, v = self.to_qkv(x)

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        q = q * self.scale

        if self.data_dependent_rel_pos and not ablate_state_transition:
            frac_gradient = self.frac_gradient_data_dependent_rel_pos

            a = self.to_a(x)

            # allow for data dependent relative position projection to change more slowly
            # alternative to using a lowered learning rate mentioned in paper

            a = a * frac_gradient + a.detach() * (1 - frac_gradient)

            a = torch.view_as_complex(a)

            if ablate_complex:
                a = a.real + 0.j

            magnitude, phase = a.abs(), a.angle()
            a = torch.polar(magnitude.sigmoid(), phase)

            a = rearrange(a, '... -> ... 1')
            a_cumprod = a.cumprod(dim = -2)

            a_cumprod_real = a_cumprod.real.clamp(min = 1e-10)
            a_cumprod_real_inverse = 1. / a_cumprod_real

            q, k = map(lambda t: rearrange(t, '... (d c) -> ... d c', c = 2), (q, k))

            q = q * a_cumprod_real
            k = k * a_cumprod_real_inverse

            q, k = map(lambda t: rearrange(t, '... d c -> ... (d c)'), (q, k))

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        i, j = sim.shape[2:]
        causal_mask = torch.ones((i, j), dtype = torch.bool, device = x.device).triu(j - i + 1)

        if self.softmax_normalize:
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)
            attn = sim.softmax(dim = -1)
        else:
            attn = sim.masked_fill(causal_mask, 0.)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        if exists(self.to_gates):
            out = out * self.to_gates(x)

        return self.to_out(out)

# data gated linear attention with "gateloop operator"

def gate_loop_operator(q, k, v, a):
    """
    the pseudocode in section 3.2 of the paper
    """

    kv = einsum('b n d, b n e -> b n d e', k, v)
    kv = kv + 0.j

    def binary_operator(a, b):
        a_i, kv_i = a
        a_j, kv_j = b
        return a_j * a_i, a_j * kv_i + kv_j

    _, kv = associative_scan(binary_operator, (a, kv))

    return einsum('b n d, b n d e -> b n e', q, kv.real)

class GateLoopedAttention(Module):
    def __init__(
        self,
        dim,
        heads = None,
        dim_inner = None,
        checkpoint_gate_looped_attn = True,
        add_swish_gating = True,
        sub_ln = False,
        frac_gradient_state_transition = 0.9
    ):
        super().__init__()
        self.frac_gradient_state_transition = frac_gradient_state_transition
        self.checkpoint_gate_looped_attn = checkpoint_gate_looped_attn

        dim_inner = default(dim_inner, dim)
        heads = default(heads, dim_inner)

        self.heads = heads
        assert (dim_inner % heads) == 0, f'dimension for gate looped attention {dim_inner} must be divisible by number of gate loop heads {heads}'

        self.split_heads = Rearrange('b n (h d) -> (b h) n d', h = heads)

        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias = False)

        self.to_a = nn.Sequential(
            nn.Linear(dim, heads * 2),
            Rearrange('b n (h c) -> (b h) n 1 1 c', h = heads, c = 2)
        )

        self.merge_heads = Rearrange('(b h) n d -> b n (h d)', h = heads)

        self.maybe_sub_ln = nn.LayerNorm(dim_inner) if sub_ln else nn.Identity()

        self.to_gates = None

        if add_swish_gating:
            self.to_gates = nn.Sequential(
                nn.Linear(dim, dim_inner, bias = False),
                nn.SiLU()
            )

        self.to_out = nn.Linear(dim_inner, dim, bias = False) if dim_inner != dim or add_swish_gating else nn.Identity()

    def forward(
        self,
        x,
        ablate_complex = False,
        ablate_state_transition = False
    ):
        frac_gradient = self.frac_gradient_state_transition

        q, k, v = self.to_qkv(x).chunk(3, dim = -1)

        q, k, v = map(self.split_heads, (q, k, v))

        a = self.to_a(x)
        a = a * frac_gradient + a.detach() * (1 - frac_gradient)

        a = torch.view_as_complex(a)

        if ablate_complex:
            a = a.real + 0.j

        if ablate_state_transition:
            a = torch.ones_like(a.real) + 0.j
        else:
            # activations for state transitions
            # sigmoid for magnitude, identity for phase

            magnitude, phase = a.abs(), a.angle()
            a = torch.polar(magnitude.sigmoid(), phase)

        need_backwards = any([t.requires_grad for t in (q, k, v, a)])

        fn = partial(checkpoint, gate_loop_operator) if need_backwards and self.checkpoint_gate_looped_attn else gate_loop_operator

        out = fn(q, k, v, a)

        out = self.merge_heads(out)

        out = self.maybe_sub_ln(out)

        if exists(self.to_gates):
            out = self.to_gates(x) * out

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
        checkpoint_gate_looped_attn = True,
        use_gate_looped_attn = True,
        gate_loop_heads = None,
        attn_add_swish_gating = True,
        dim_gate_looped_attn = None,
        attn_softmax_normalize = None,
        data_dependent_rel_pos = False,
        frac_gradient_state_transition = 0.9,
        ablate_complex = False,
        ablate_state_transition = False,
        rotary_emb = False,
        post_ln_norm = False,
        sub_ln = False
    ):
        super().__init__()
        self.ablate_complex = ablate_complex
        self.ablate_state_transition = ablate_state_transition

        self.token_emb = nn.Embedding(num_tokens, dim)

        layers = ModuleList([])

        layer_wrapper = PreNorm if not post_ln_norm else PostNorm

        for _ in range(depth):

            if use_gate_looped_attn:
                spatial_mixer = GateLoopedAttention(
                    dim = dim,
                    heads = gate_loop_heads,
                    dim_inner = dim_gate_looped_attn,
                    add_swish_gating = attn_add_swish_gating,
                    sub_ln = sub_ln,
                    checkpoint_gate_looped_attn = checkpoint_gate_looped_attn,
                    frac_gradient_state_transition = frac_gradient_state_transition
                )
            else:
                spatial_mixer = CausalFullAttention(
                    dim = dim,
                    dim_head = dim_head,
                    heads = heads,
                    rotary_emb = rotary_emb,
                    add_swish_gating = attn_add_swish_gating,
                    softmax_normalize = attn_softmax_normalize,
                    data_dependent_rel_pos = data_dependent_rel_pos,
                    frac_gradient_data_dependent_rel_pos = frac_gradient_state_transition
                )

            channelwise_mixer = FeedForward(
                dim = dim,
                mult = ff_mult
            )

            layers.append(ModuleList([
                layer_wrapper(dim, spatial_mixer),
                layer_wrapper(dim, channelwise_mixer)
            ]))

        self.layers = ModuleList(layers)

        self.to_logits = Sequential(
            RMSNorm(dim) if not post_ln_norm else None,
            nn.Linear(dim, num_tokens, bias = False)
        )

    def forward(
        self,
        x,
        return_loss = False,
        ablate_complex = None,
        ablate_state_transition = None
    ):
        ablate_complex = default(ablate_complex, self.ablate_complex)
        ablate_state_transition = default(ablate_state_transition, self.ablate_state_transition)

        if return_loss:
            x, labels = x[:, :-1], x[:, 1:]

        x = self.token_emb(x)

        for attn, ff in self.layers:
            x = attn(
                x,
                ablate_complex = ablate_complex,
                ablate_state_transition = ablate_state_transition
            )

            x = ff(x)

        logits = self.to_logits(x)

        if not return_loss:
            return logits

        logits = rearrange(logits, 'b n c -> b c n')
        return F.cross_entropy(logits, labels)
