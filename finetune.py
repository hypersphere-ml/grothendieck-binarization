"""Fine-tune one method at one bit width b (Table 3): up to 1,600 steps of AdamW (betas 0.9 and 0.95, no weight decay)
on the 48 body weight matrices only, without dropout, gradient norm clipped at 1, batches of 48 windows of 1024 tokens
under bf16 autocast; the learning rate rises linearly over 200 steps and then follows a cosine schedule down to a tenth
of its peak. Validation partition A is scored at the start and every 50 steps; the state with the best perplexity on A,
the starting weights included, is kept; the run stops after ten evaluations in a row without a 0.1% improvement; the
kept state is scored on partition B and the test split. --steps 0 scores the method without fine-tuning; --steps 6400
runs the extra comparison at four times the budget (README), with warmup 400, evaluations every 100 steps, and misses
counted only at evaluations where the rate is at most half its peak (step 100, and from step 3,700 on).

    python finetune.py --method ours --b 1 --lr 1e-3 --seed 1337

G (ours) or R (rotated) is drawn with seed + 100 and the batches with seed + 1001, as in the reported runs.
"""
import argparse
import json
import math
from pathlib import Path

import torch

import data
import layers
import model as gpt

SCHEDULE = {1600: (200, 50), 6400: (400, 100)}       # steps: (warmup steps, evaluation interval)
PATIENCE, MIN_IMPROVEMENT = 10, 0.001
MICRO_BATCH, ACCUMULATION = 12, 4


def lr_multiplier(step, warmup, steps):
    """Linear warmup, then cosine decay to 0.1 of the peak rate."""
    if step < warmup:
        return step / warmup
    return 0.1 + 0.45 * (1.0 + math.cos(math.pi * min(1.0, (step - warmup) / (steps - warmup))))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=("ours", "scalar", "rotated"), required=True,
                   help="ours: hypersphere binarization with Hadamard blocks; scalar or rotated scalar quantization")
    p.add_argument("--b", type=int, required=True, help="about b bits per weight (for ours, N = b n)")
    p.add_argument("--lr", type=float, default=0.0, help="the peak learning rate, required when --steps > 0")
    p.add_argument("--seed", type=int, default=1337,
                   help="draws G or R with seed + 100, the batches with seed + 1001 (default %(default)s)")
    p.add_argument("--steps", type=int, choices=(0, *SCHEDULE), default=1600,
                   help="0: no fine-tuning; 1600: the paper; 6400: the extra runs (default %(default)s)")
    p.add_argument("--baseline", default="baseline.pt", help="the baseline checkpoint (default %(default)s)")
    p.add_argument("--data", default="data", help="the token cache (default %(default)s)")
    p.add_argument("--out", default="runs/finetune", help="the directory of the run records (default %(default)s)")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    if a.steps and not a.lr > 0:
        p.error("--lr must be positive when --steps > 0")

    splits = data.load(a.data)
    val_a, val_b = data.validation_partitions(splits["validation"])
    model = gpt.load(a.baseline, a.device)
    if a.method == "ours":
        layers.binarize(model, "hadamard", a.b, a.seed + 100)
    else:
        layers.quantize(model, a.b, a.seed + 100 if a.method == "rotated" else None)
    weights = [layer.weight for _, layer in gpt.body(model)]
    model.requires_grad_(False)
    for w in weights:
        w.requires_grad_(True)

    best = data.perplexity(model, val_a, a.device)
    curve, best_step, anchor, misses, stopped = [[0, best]], 0, best, 0, "budget"
    kept = [w.detach().clone() for w in weights]
    if a.steps:
        warmup, interval = SCHEDULE[a.steps]
        optimizer = torch.optim.AdamW(weights, lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
        generator = torch.Generator().manual_seed(a.seed + 1001)
        for step in range(1, a.steps + 1):
            model.train()
            multiplier = lr_multiplier(step, warmup, a.steps)
            optimizer.param_groups[0]["lr"] = a.lr * multiplier
            optimizer.zero_grad(set_to_none=True)
            for _ in range(ACCUMULATION):
                x, y = data.batch(splits["train"], MICRO_BATCH, generator, a.device)
                with torch.autocast(torch.device(a.device).type, dtype=torch.bfloat16):
                    loss = model(x, y)
                (loss / ACCUMULATION).backward()
            torch.nn.utils.clip_grad_norm_(weights, 1.0)
            optimizer.step()
            if step % interval != 0 and step < a.steps:
                continue
            ppl = data.perplexity(model, val_a, a.device)
            curve.append([step, ppl])
            print(f"step {step}: validation A {ppl:.3f}", flush=True)
            if ppl < best:
                best, best_step, kept = ppl, step, [w.detach().clone() for w in weights]
            if ppl < anchor * (1 - MIN_IMPROVEMENT):
                anchor, misses = ppl, 0
            elif a.steps == 1600 or multiplier <= 0.5:
                misses += 1
            if misses >= PATIENCE:
                stopped = f"patience at step {step}"
                break
        with torch.no_grad():
            for w, k in zip(weights, kept):
                w.copy_(k)

    result = dict(method=a.method, b=a.b, lr=a.lr, steps=a.steps, seed=a.seed, curve=curve, selected_step=best_step,
                  stopped=stopped, val_a=best, val_b=data.perplexity(model, val_b, a.device),
                  test=data.perplexity(model, splits["test"], a.device))
    print({k: v for k, v in result.items() if k != "curve"})
    Path(a.out).mkdir(parents=True, exist_ok=True)
    name = f"{a.method}_b{a.b}_lr{a.lr:g}_steps{a.steps}_seed{a.seed}.json"
    (Path(a.out) / name).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
