<img src="./gateloop.png" width="450px"></img>

## GateLoop Transformer

Implementation of <a href="https://arxiv.org/abs/2311.01927">GateLoop</a> Transformer in Pytorch and Jax, to be tested on Enwik8 character level modeling.

Update: A transformer run with regular attention + data dependent xpos relative positions did not converge at all. Also, gate loop's associative scan also is not able to train on even sequence lengths of 128. I'm not sure if it can be done without a specialized CUDA kernel, much like autoregressive linear attention (RWKV and the like)

Update 2: Got a smaller GateLoop transformer (gate loop dimensions of 128) to run on sequence length of 256. It is converging very well with a quick eyeball. Will run some more rigorous experiments tomorrow.

Update 3: Fixed a misunderstanding and definitely seems to be converging better than vanilla linear attention (from my memories of those experiments).

Update 4: <a href="https://api.wandb.ai/links/lucidrains/ysbz84fn">Ongoing experiments</a>

Update 5: Author has reviewed the code, and there was another misunderstanding. They use maximum heads (heads == dimension). This is kind of a plot twist, as this is infeasible for normal attention. It also obviates the need a fused CUDA kernel as in autoregressive linear attention.

Update 6: Corrected gateloop transformer run looks amazing. Cautiously optimistic now.

Update 7: Ablating state transition shows expected negative result. Ablating complex valued states though, I see no difference, at least, early in the run.

Update 8: Directly projecting to `kv` with one projection for the max-heads setting (instead of keys and values separately followed by element-wise multiplication) yields similar results

Update 9: <a href="https://api.wandb.ai/links/lucidrains/do1i9rx0">Head to head to 20k</a>, just to make sure Gateloop doesn't get exceeded later on

Update 10: and it got passed by attention, at least, assuming the implementation in the repo is correct.

Update 11: I'm seeing a steady improvement increasing the head dimension, so I no longer believe max-heads is optimal. Increasing the head dimension brings us right back to linear attention and needing the fused CUDA kernel.

Update 12: <a href="https://github.com/cnapun">Nikil</a> spotted a potential error with the `kv` not being kept in complex (and real component taken at end). <a href="https://api.wandb.ai/links/lucidrains/lgz368mf">Rerunning experiments</a>

Update 13: Still clearly worse

Update 14: See some synergy when mixing gateloop and attention on a small scale, when holding parameters constant. Will be adding a tiny bit of simplified gateloop layers to transformers to address a main weakness in attention for future projects.

Update 15: There may be a way to combine associative scan based works with the findings from the recently proposed <a href="https://arxiv.org/abs/2312.04927">taylor series linear attention</a>. will carry out some independent research before end of January 2024 and share the results here.

### Appreciation

- <a href="https://stability.ai/">StabilityAI</a>, <a href="https://a16z.com/supporting-the-open-source-ai-community/">A16Z Open Source AI Grant Program</a>, and <a href="https://huggingface.co/">🤗 Huggingface</a> for the generous sponsorships, as well as my other sponsors, for affording me the independence to open source current artificial intelligence research

### Install

```bash
$ pip install gateloop-transformer
```

### Usage

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

A simplified gate loop layer

```python
import torch
from gateloop_transformer import SimpleGateLoopLayer

gateloop = SimpleGateLoopLayer(512)

x = torch.randn(1, 65536, 512)
x = gateloop(x) + x
```
### Character-level Language Modeling

Install requirements

```bash
$ pip install -r requirements.txt
```

Then run the `train.py` script for autoregressive modeling on enwik8

```bash
$ python train.py
```

### Todo

- [x] jax version with equinox
- [x] start with naive memory checkpointing of gate loop operation
- [x] retry the failed full attention experiments (with data dependent xpos), but with complex valued scales (didn't work)
- [x] separate out a minimal gateloop circuit, to augment attention, rather than to replace it, as done in <a href="https://arxiv.org/abs/2209.10655">Mega</a>
- [x] experiments
    - [x] do all the ablations and figure out how much the data controlled state transitions adds (as well as whether it needs to be complex)
    - [x] do complete runs between transformer + rotary against gateloop with max heads, parameter held constant to 20k steps
- [x] just use jax's associative scan, wrapped with jax2torch, for now. pytorch team claim they will implement <a href="https://github.com/pytorch/pytorch/issues/95408">this</a> eventually

## Citations

```bibtex
@inproceedings{Katsch2023GateLoopFD,
    title   = {GateLoop: Fully Data-Controlled Linear Recurrence for Sequence Modeling},
    author  = {Tobias Katsch},
    year    = {2023},
    url     = {https://api.semanticscholar.org/CorpusID:265018962}
}
```

```bibtex
@inproceedings{Heinsen2023EfficientPO,
    title   = {Efficient Parallelization of a Ubiquitous Sequential Computation},
    author  = {Franz A. Heinsen},
    year    = {2023},
    url     = {https://api.semanticscholar.org/CorpusID:265213659}
}
```
