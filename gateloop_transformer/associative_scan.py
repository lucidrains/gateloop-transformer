# taken from S5-pytorch repository
# https://github.com/i404788/s5-pytorch/blob/74e2fdae00b915a62c914bf3615c0b8a4279eb84/s5/jax_compat.py#L51-L134

# will be adapted to test out GateLoop on a small scale https://arxiv.org/abs/2311.01927

import torch
from torch import Tensor
import torch.nn.functional as F

from typing import Tuple, Callable

# helper functions

def safe_map(f, *args):
    args = list(map(list, args))
    n = len(args[0])
    for arg in args[1:]:
        assert len(arg) == n, f'length mismatch: {list(map(len, args))}'
    return list(map(f, *args))

def pad_at_dim(t, pad, dim = -1, value = 0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)

# Pytorch impl. of jax.lax.associative_scan
# made specifically for axis of 1 (sequence of tokens for autoregressive modeling)

def associative_scan(
    operator: Callable,
    elems: Tuple[Tensor, Tensor]
):
    num_elems = int(elems[0].shape[1])

    if not all(int(elem.shape[1]) == num_elems for elem in elems[1:]):
        raise ValueError('Array inputs to associative_scan must have the same '
                         'first dimension. (saw: {})'
                         .format([elem.shape for elem in elems]))

    def _scan(elems):
        """Perform scan on `elems`."""
        num_elems = elems[0].shape[1]

        if num_elems < 2:
            return elems

        # Combine adjacent pairs of elements.

        reduced_elems = operator(
          [elem[:, :-1:2] for elem in elems],
          [elem[:, 1::2] for elem in elems])

        # Recursively compute scan for partially reduced tensors.

        odd_elems = _scan(reduced_elems)

        if num_elems % 2 == 0:
            even_elems = operator(
                [e[:, :-1] for e in odd_elems],
                [e[:, 2::2] for e in elems])
        else:
            even_elems = operator(
                odd_elems,
                [e[:, 2::2] for e in elems])

        # The first element of a scan is the same as the first element
        # of the original `elems`.

        even_elems = [
          torch.cat([elem[:, :1], result], dim=1)
          for (elem, result) in zip(elems, even_elems)]

        return list(safe_map(_interleave, even_elems, odd_elems))

    return _scan(elems)

def _interleave(a, b):
    a_axis_len, b_axis_len = a.shape[1], b.shape[1]
    output_axis_len = a_axis_len + b_axis_len

    if (a_axis_len == (b_axis_len + 1)):
        b = pad_at_dim(b, (0, 1), dim = 1)

    stacked = torch.stack([a, b], dim=2)
    interleaved = torch.flatten(stacked, start_dim=1, end_dim=2)

    return interleaved[:, :output_axis_len]
