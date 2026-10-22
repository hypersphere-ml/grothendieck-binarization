# Grothendieck identity hypersphere binarization

Code for *Theory and applications of Grothendieck identity hypersphere binarization*
([arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)): our binarized linear layer, the scripts behind every figure and
table of the paper, and the results of every run reported there.

The layer approximates a linear layer $x \mapsto Wx$, with $W$ an $m \times n$ matrix, by

```math
f_G(x; W) = \|x\| \, D \sin\Bigl(\frac{\pi}{2N} \mathrm{sign}(W G^\top) \mathrm{sign}(G x)\Bigr),
```

where $D$ is the diagonal matrix of the row norms of $W$, $G$ is a random $N \times n$ matrix, and $\sin$ and
$\mathrm{sign}$ act entrywise (Section 3.1). Deployed, the layer stores the $m \times N$ signs
$\mathrm{sign}(W G^\top)$, which is $N/n$ bits per weight, the $m$ row norms, and the $N$ random signs that
define a Hadamard-block $G$.

## Checking the reported numbers

```bash
python tables.py        # Tables 1-3, from the run records in results/
```

This needs only Python: no packages, data, checkpoint or GPU. Perplexities are shown as ratios to the baseline's test
perplexity, with the mean and the sample standard deviation over draws of $G$ or $R$, or over seeds. It also prints one
comparison that is not in the paper, [a longer fine-tuning budget](#extra-not-in-the-paper-a-longer-fine-tuning-budget).

## Using the layer

```python
from layers import CONSTRUCTIONS, HypersphereLinear

linear = model.fc                    # a trained nn.Linear; n = linear.in_features
# N = 2n rows of Hadamard blocks: 2 bits per weight
G = CONSTRUCTIONS["hadamard"](linear.in_features, 2, linear.weight.device)
model.fc = HypersphereLinear(linear, G)
```

The constructions of $G$ are `hadamard` (Hadamard blocks; $n$ a power of 2, or 12 times one), `orthogonal` (orthogonal
blocks), `gaussian` and `rademacher`. The code simulates the binarized and quantized layers in floating point; it does
not pack the bits.

## Setup

```bash
pip install -r requirements.txt
```

We used Python 3.14, PyTorch 2.13 and one NVIDIA RTX 5090 (32 GB); training the baseline peaks at about 15 GB of GPU
memory. On first use, `train_baseline.py`, `direct.py` and `finetune.py` download `wikitext-103-raw-v1` from Hugging
Face and write its GPT-2 tokens to `data/`. Run the commands from the repository root.

## Reproducing the paper

Each run of `train_baseline.py`, `direct.py` or `finetune.py` writes its results to a JSON file under `runs/`
(`runs/baseline.json`, `runs/direct/`, `runs/finetune/`), the same layout as `results/`. To print the tables from your
runs instead of ours:

```bash
python tables.py runs
```

### Protocol

- Perplexity is computed over consecutive non-overlapping 1024-token windows, each scored without earlier context.
- The first 120 windows of the validation split form validation partition A and the remaining 125 form partition B.
  A selects the kept state of a training or fine-tuning run, B selects hyperparameters, and only test perplexities are
  reported.
- The experiments binarize or quantize the 48 linear layers of the transformer blocks; the token and position
  embeddings (the output layer is tied to the token embedding), the biases and the layer norms stay in floating point.

### Arguments

Every script lists its arguments and defaults with `--help`. Those in the commands below:

| script | argument | meaning |
|---|---|---|
| `direct.py` | `--construction` | the construction of $G$: `hadamard`, `orthogonal`, `gaussian` or `rademacher` |
| | `--b` | $N = bn$ |
| | `--seed` | the draw of $G$ |
| `finetune.py` | `--method` | the column of Table 3: `ours` (our hypersphere binarization with Hadamard blocks), `scalar` (scalar quantization) or `rotated` (rotated scalar quantization) |
| | `--b` | about $b$ bits per weight (for `ours`, $N = bn$) |
| | `--lr` | the peak learning rate |
| | `--steps` | `0` (no fine-tuning), `1600` (the paper; the default) or `6400` (the longer budget, not in the paper) |
| | `--seed` | $G$ or $R$ is drawn with seed + 100 and the batches with seed + 1001 (default 1337) |
| `train_baseline.py` | `--evaluate` | score this checkpoint instead of training one |
| | `--out` | where to save the trained checkpoint |
| `direct.py`, `finetune.py` | `--baseline` | the baseline checkpoint (default `baseline.pt`) |

### Baseline

Download `baseline.pt` (498 MB, sha256 `cbe0114fb5c91cb8e3101bccb288194a6d9a73c44c64260d2bf9c3210b14404a`) from the
[release](https://github.com/hypersphere-ml/grothendieck-binarization/releases) into the repository root, then

```bash
python train_baseline.py --evaluate baseline.pt     # validation A 14.48, validation B 17.65, test 16.37
```

which also writes `runs/baseline.json`.

To train a baseline instead (about 8 hours on an RTX 5090):

```bash
python train_baseline.py --out my_baseline.pt      # also writes runs/baseline.json
```

then pass `--baseline my_baseline.pt` to `direct.py` and `finetune.py`. The recipe:

- GPT-2 small from random initialization, 80,000 steps of AdamW (betas 0.9 and 0.95, weight decay 0.3 on the matrices
  and embeddings), dropout 0.1, gradient norm clipped at 1, bfloat16 autocast;
- batches of 48 windows of 1024 tokens drawn at random from the training split;
- learning rate rising linearly to 3e-4 over 500 steps, then a cosine schedule to zero;
- the kept state is the one with the best perplexity on partition A, scored every 500 steps (ours: step 77,500);
- dropout and weight decay chosen from a small grid by perplexity on partition B.

### Figures 1-3

The scripts write `figures/fig00a.eps`, `fig00b.eps`, `fig01.eps` and `fig02.eps`. Figures 2 and 3 need LaTeX, dvips
and Ghostscript (matplotlib's `usetex`).

```bash
python figures/fig00.py
python figures/fig01.py
python figures/fig02.py
```

### Tables 1 and 2

Each run takes under a minute on an RTX 5090.

```bash
for c in hadamard orthogonal gaussian rademacher; do
  for b in 1 2 3 6; do
    for s in 1337 1338 1339 1340 1341; do
      python direct.py --construction $c --b $b --seed $s
    done
  done
done
```

### Table 3 without fine-tuning

Our method's column is Table 1's Hadamard column; scalar quantization is deterministic; rotated scalar quantization is
averaged over five draws of $R$.

```bash
for b in 1 2 3 6; do
  python finetune.py --method scalar --b $b --steps 0
  for s in 1337 1338 1339 1340 1341; do python finetune.py --method rotated --b $b --steps 0 --seed $s; done
done
```

### Table 3 with fine-tuning

Each run takes 12-100 minutes on an RTX 5090:

```bash
for s in 1337 1338 1339; do
  python finetune.py --method ours --b 1 --lr 1e-3 --seed $s      # and so on for each cell of the table below
done
```

- A run fine-tunes the 48 weight matrices for up to 1,600 steps and keeps the state with the best perplexity on
  partition A, scored at the start and every 50 steps; the full protocol is in `finetune.py`.
- For each method and bit width we chose the learning rate at seed 1337 by the kept state's perplexity on partition B,
  from the grid {1, 3} x 10^k extended until the chosen rate was at neither end of the rates tried, or the lowest rate
  when every run kept its starting weights. Seeds 1338 and 1339 were then run at the chosen rate.
- `results/finetune` holds every rate we tried; `tables.py` applies the rule to the runs it finds and marks a chosen
  rate at an end of them with `!`.

The chosen learning rates, by bit width b:

| b | ours | scalar | rotated |
|---|---|---|---|
| 1 | 1e-3 | 1e-3 | 1e-3 |
| 2 | 1e-3 | 3e-4 | 3e-4 |
| 3 | 3e-4 | 3e-4 | 3e-5 |
| 6 | 3e-4 | 3e-6 | 1e-7 |

At six bits every rotated scalar run at seed 1337 kept its starting weights, so the rule gives the lowest rate; at 1e-7
two of the three seeds keep their starting weights and the third keeps its state at step 50.

### Extra, not in the paper: a longer fine-tuning budget

At 1,600 steps every one- and two-bit run keeps one of its last two evaluations, so we also ran the one-bit cells at
seed 1337 for 6,400 steps (44-139 minutes each on an RTX 5090; warmup, evaluation interval and patience adjusted as
`finetune.py` states). `tables.py` prints them against the 1,600-step runs.

```bash
for m in ours scalar rotated; do python finetune.py --method $m --b 1 --lr 1e-3 --steps 6400 --seed 1337; done
```

## The paper mapped to the code

| paper | code |
|---|---|
| Figure 1 | [`figures/fig00.py`](figures/fig00.py) |
| Figures 2 and 3; the error $d_{G,W,x}$ of (2.1) | [`figures/fig01.py`](figures/fig01.py), [`figures/fig02.py`](figures/fig02.py); `errors` in [`figures/common.py`](figures/common.py) |
| the constructions of $G$ (Section 2; Hadamard blocks as in Section 3.3) | `CONSTRUCTIONS` in [`layers.py`](layers.py) |
| $f_G$, (3.1), (3.2) and (3.4), with the surrogate of Section 3.5 | `HypersphereLinear` in [`layers.py`](layers.py) |
| scalar and rotated scalar quantization, straight-through gradients (Section 3.5) | `round_scalar`, `ScalarLinear`, `quantize` in [`layers.py`](layers.py) |
| the baseline (Section 3.2); the 48 linear layers of the transformer blocks | `GPT`, `body` in [`model.py`](model.py); [`train_baseline.py`](train_baseline.py) |
| perplexity and the validation partitions | `perplexity`, `validation_partitions` in [`data.py`](data.py) |
| Tables 1 and 2; the scaled error (3.3) | [`direct.py`](direct.py) |
| fine-tuning and the choice of learning rate (Section 3.5); Table 3 | [`finetune.py`](finetune.py), `choose_lr` in [`tables.py`](tables.py) |
| Tables 1-3 | [`results/`](results), printed by [`tables.py`](tables.py) |

## Notes

- With the released checkpoint, `train_baseline.py --evaluate`, `direct.py` and `finetune.py --steps 0` reproduce
  `results/` exactly. $G$ and $R$ come from the GPU's random number generator, so a seed gives the paper's draws only on
  a CUDA device.
- Fine-tuning is not deterministic on a GPU. Reruns of the one-bit cells of our method and of scalar quantization at
  seed 1337 ended 0.6% and 2.3% from their results in `results/`, within each cell's spread over seeds.
- Training runs under bfloat16 autocast; every perplexity is computed in float32.
- `torch.sign(0)` is 0, where the paper takes sign(0) = 1. On validation data, fewer than 1 in 10,000 of the entries it
  receives are exactly zero (GELU outputs that round to zero, and bfloat16 rounding in training); the code keeps torch's
  convention, as the reported runs did.
- The last test article, "The Hustler (film)", is also in the WikiText-103 training split. It fills the last three of
  the 280 test windows; without them the baseline's test perplexity is 16.376 instead of 16.371.

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
