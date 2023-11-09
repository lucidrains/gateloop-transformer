<img src="./gateloop.png" width="450px"></img>

## GateLoop Transformer (wip)

Implementation of <a href="https://arxiv.org/abs/2311.01927">GateLoop</a> Transformer in both Jax and Pytorch, to be tested on Enwik8 character level modeling.

Jax version will be done with the <a href="https://github.com/patrick-kidger/equinox">Equinox</a> framework

Update: A transformer run with regular attention + data dependent xpos relative positions did not converge at all. Also, gate loop's associative scan also is not able to train on even sequence lengths of 128. I'm not sure if it can be done without a specialized CUDA kernel, much like autoregressive linear attention (RWKV and the like)

## Install

```bash
$ pip install gateloop-transformr
```

## Usage

```python
import torch
from gateloop_transformer import Transformer

model = Transformer(
    num_tokens = 256,
    dim = 624,
    depth = 6,
    use_gate_looped_attn = True
)

ids = torch.randint(0, 256, (1, 1024))
logits = model(ids) # (1, 1024, 256)
```

## Citations

```bibtex
@inproceedings{Katsch2023GateLoopFD,
    title   = {GateLoop: Fully Data-Controlled Linear Recurrence for Sequence Modeling},
    author  = {Tobias Katsch},
    year    = {2023},
    url     = {https://api.semanticscholar.org/CorpusID:265018962}
}
```
