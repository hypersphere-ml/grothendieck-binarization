# Grothendieck identity hypersphere binarization

Code for *Theory and applications of Grothendieck identity hypersphere binarization*
([arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)): our binarized linear layer, the scripts behind every figure and
table of the paper, and the results of every run reported there.

The layer approximates a linear layer x -> Wx, with W an m x n matrix, by

    f_G(x; W) = ||x|| D sin(pi/(2N) sign(W G^T) sign(G x)),

where D = diag(||w_1||, ..., ||w_m||) holds the row norms of W, G is a random N x n matrix, and sin and sign act entrywise
(Section 3.1). Deployed, the layer stores the signs sign(W G^T), about N/n bits per weight, and the m row norms.

## Using the layer

```python
from layers import CONSTRUCTIONS, HypersphereLinear

linear = model.fc                                    # a trained nn.Linear of your model, with n = linear.in_features
G = CONSTRUCTIONS["hadamard"](linear.in_features, 2, linear.weight.device)   # N = 2n rows: about 2 bits per weight
model.fc = HypersphereLinear(linear, G)
```

The constructions of G are `hadamard` (Hadamard blocks; n a power of 2, or 12 times one), `orthogonal` (orthogonal
blocks), `gaussian` and `rademacher`. In the experiments, the 48 linear layers of the transformer blocks are binarized or
quantized; the token and position embeddings (the output layer shares the token embedding), the biases and the layer norms
stay in floating point. The code simulates the binarized and quantized layers in floating point; it does not pack the bits.

## The paper in the code

| paper | code |
|---|---|
| Figures 1, 2 and 3; the error d_{G,W,x}, (2.1) | `figures/fig00.py`, `fig01.py`, `fig02.py`; `errors` in `figures/common.py` |
| the constructions of G (Section 2; Hadamard blocks as in Section 3.3) | `layers.CONSTRUCTIONS` |
| (3.1) | `layers.HypersphereLinear.estimate` |
| f_G, (3.2) and (3.4), with the surrogate of Section 3.5 | `layers.HypersphereLinear` |
| the 48 linear layers of the transformer blocks | `model.body` |
| the baseline (Section 3.2) | `model.GPT`, `train_baseline.py` |
| Tables 1 and 2; the scaled error (3.3) on the first 16 validation windows | `direct.py`, `direct.scaled_error` |
| scalar and rotated scalar quantization, straight-through gradients (Section 3.5) | `layers.round_scalar`, `layers.ScalarLinear`, `layers.quantize` |
| fine-tuning and the choice of learning rate (Section 3.5), Table 3 | `finetune.py`, `tables.choose_lr` |
| perplexity (consecutive non-overlapping 1024-token windows) and the validation partitions | `data.perplexity`, `data.validation_partitions` |
| every number in Tables 1-3 (and the extra longer-budget comparison below) | `results/`, printed by `tables.py` |

## Setup

```
pip install -r requirements.txt
```

We used Python 3.14, PyTorch 2.13 and one NVIDIA RTX 5090 (32 GB); training the baseline peaks at about 15 GB of GPU
memory. Figures 2 and 3 need LaTeX, dvips and Ghostscript (matplotlib's `usetex`). The first script that loads data
downloads `wikitext-103-raw-v1` from Hugging Face and writes its GPT-2 tokens to `data/` (240 MB; building them takes
about a minute and 10 GB of memory). Run the commands from the repository root.

Download `baseline.pt` (498 MB, sha256 `cbe0114fb5c91cb8e3101bccb288194a6d9a73c44c64260d2bf9c3210b14404a`) from the
[release](https://github.com/hypersphere-ml/grothendieck-binarization/releases) into the repository root, then

```
python train_baseline.py --evaluate baseline.pt     # validation A 14.48, validation B 17.65, test 16.37
```

## Reproducing the paper

`python tables.py` prints Tables 1-3 from `results/`. `train_baseline.py`, `direct.py` and
`finetune.py` write one JSON file per run under `runs/`, laid out as `results/` is, and `python tables.py runs` prints the
tables from your runs (after `train_baseline.py --evaluate baseline.pt` has written `runs/baseline.json`).

**Figures 1-3** (about 1 s, 14 min and 2 min) write `figures/fig00a.eps`, `fig00b.eps`, `fig01.eps` and `fig02.eps`:

```
python figures/fig00.py
python figures/fig01.py
python figures/fig02.py
```

**Baseline.** `python train_baseline.py --out my_baseline.pt --record my_baseline.json` (about 8 hours). It trains GPT-2
small from random initialization for 80,000 steps of AdamW (betas 0.9 and 0.95, weight decay 0.3 on the matrices and
embeddings) on batches of 48 windows of 1024 tokens drawn at random from the training split, with dropout 0.1, the
gradient norm clipped at 1 and bfloat16 autocast; the learning rate rises linearly to 3e-4 over 500 steps and decays to
zero by a cosine. It keeps the state with the best perplexity on validation partition A, scored every 500 steps (ours:
step 77,500). Dropout and weight decay were chosen from a small grid by perplexity on validation partition B.

**Tables 1 and 2.** `--b` sets N = bn. Each run takes under a minute:

```
for c in hadamard orthogonal gaussian rademacher; do for b in 1 2 3 6; do for s in 1337 1338 1339 1340 1341; do
  python direct.py --construction $c --b $b --seed $s
done; done; done
```

**Table 3 without fine-tuning.** Our method's column is Table 1's Hadamard column; scalar quantization is deterministic;
rotated scalar quantization is averaged over five draws of R (`finetune.py` draws R with seed + 100):

```
for b in 1 2 3 6; do
  python finetune.py --method scalar --b $b --steps 0
  for s in 1337 1338 1339 1340 1341; do python finetune.py --method rotated --b $b --steps 0 --seed $s; done
done
```

**Table 3 with fine-tuning.** Each run takes 12-100 minutes. For each method and bit width we chose the learning rate at
seed 1337 by the kept weights' perplexity on validation partition B, from the grid {1, 3} x 10^k extended until the chosen
rate was at neither end of the rates tried (the lowest rate when no rate improved on the starting weights);
`results/finetune` holds every rate we tried, and `tables.py` applies this rule to the runs it finds. Seeds 1338 and 1339
then ran at the chosen rate:

| b | ours | scalar | rotated |
|---|---|---|---|
| 1 | 1e-3 | 1e-3 | 1e-3 |
| 2 | 1e-3 | 3e-4 | 3e-4 |
| 3 | 3e-4 | 3e-4 | 3e-5 |
| 6 | 3e-4 | 3e-6 | 1e-7 |

At six bits no rate improved rotated scalar quantization's starting weights at seed 1337, so the rule gives the lowest
rate; at 1e-7 two of the three seeds keep the starting weights and the third keeps step 50.

```
for s in 1337 1338 1339; do
  python finetune.py --method ours --b 1 --lr 1e-3 --seed $s      # and so on for each cell of the table
done
```

**Extra, not in the paper: a longer fine-tuning budget.** At 1,600 steps every one- and two-bit run keeps one of its
last two evaluations, so we also ran the one-bit cells at seed 1337 for 6,400 steps (44-139 minutes each; warmup,
evaluation interval and patience scaled as `finetune.py` states). `tables.py` prints them against the 1,600-step runs.

```
for m in ours scalar rotated; do python finetune.py --method $m --b 1 --lr 1e-3 --steps 6400 --seed 1337; done
```

## Notes

- With the released checkpoint, `train_baseline.py --evaluate`, `direct.py` and `finetune.py --steps 0` reproduce
  `results/` exactly (we checked on an RTX 5090 with PyTorch 2.13). G and R come from the GPU's random number generator, so a
  seed gives the paper's draws only on a CUDA device.
- Fine-tuning is not deterministic on a GPU: two runs of the same cell differ. Reruns of the one-bit cells of our method
  and of scalar quantization at seed 1337 ended 0.6% and 2.3% from their results in `results/`, within each cell's spread
  over seeds.
- Training runs under bfloat16 autocast; every perplexity is computed in float32.
- `torch.sign(0)` is 0, where the paper takes sign(0) = 1. On validation data an exact zero reaches it in fewer than 1 in
  10,000 entries (GELU outputs that round to zero, and bfloat16 rounding in training); the code keeps torch's convention,
  as the reported runs did.
- The last test article, "The Hustler (film)", is also in the WikiText-103 training split. It fills the last three of the
  280 test windows; without them the baseline's test perplexity is 16.376 instead of 16.371.

## Citation

```bibtex
@article{grothendieck2026binarization,
  title   = {Theory and applications of {G}rothendieck identity hypersphere binarization},
  author  = {Anonymous},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

The code is released under the MIT License (`LICENSE`).
