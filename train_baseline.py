"""Train the floating-point baseline (Section 3.2): GPT-2 small from random initialization on WikiText-103, AdamW with weight
decay on the matrices and embeddings, batches of 48 windows of 1024 tokens under bf16 autocast, a 500-step linear warmup
and cosine decay to zero; the state with the best perplexity on validation partition A (scored every 500 steps) is kept.

    python train_baseline.py --out baseline.pt
    python train_baseline.py --evaluate baseline.pt
"""
import argparse
import json
import math
from pathlib import Path

import torch

import data
import model as gpt

MICRO_BATCH, ACCUMULATION, EVAL_INTERVAL = 12, 4, 500


def lr_multiplier(step, warmup, steps):
    """Linear warmup, then cosine decay to zero."""
    if step < warmup:
        return step / warmup
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, (step - warmup) / (steps - warmup))))


def evaluate(model, splits, device):
    val_a, val_b = data.validation_partitions(splits["validation"])
    return dict(val_a=data.perplexity(model, val_a, device), val_b=data.perplexity(model, val_b, device),
                test=data.perplexity(model, splits["test"], device))


def train(a, splits):
    """The trained model and the step of the kept state."""
    val_a, _ = data.validation_partitions(splits["validation"])
    torch.manual_seed(a.seed)
    model = gpt.GPT(a.dropout).to(a.device)
    params = list(model.parameters())
    optimizer = torch.optim.AdamW([{"params": [q for q in params if q.dim() >= 2], "weight_decay": a.weight_decay},
                                   {"params": [q for q in params if q.dim() < 2], "weight_decay": 0.0}],
                                  lr=a.lr, betas=(0.9, 0.95))
    generator = torch.Generator().manual_seed(a.seed)
    best, best_step, kept = math.inf, 0, None
    for step in range(1, a.steps + 1):
        model.train()
        for group in optimizer.param_groups:
            group["lr"] = a.lr * lr_multiplier(step, a.warmup, a.steps)
        optimizer.zero_grad(set_to_none=True)
        for _ in range(ACCUMULATION):
            x, y = data.batch(splits["train"], MICRO_BATCH, generator, a.device)
            with torch.autocast(torch.device(a.device).type, dtype=torch.bfloat16):
                loss = model(x, y)
            (loss / ACCUMULATION).backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        if step % EVAL_INTERVAL != 0 and step < a.steps:
            continue
        ppl = data.perplexity(model, val_a, a.device)
        print(f"step {step}: validation A {ppl:.3f}", flush=True)
        if ppl < best:
            best, best_step = ppl, step
            kept = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(kept)
    return model, best_step


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="baseline.pt")
    p.add_argument("--evaluate", help="score this checkpoint instead of training one")
    p.add_argument("--record", default="runs/baseline.json")
    p.add_argument("--steps", type=int, default=80_000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=500)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=2001)
    p.add_argument("--data", default="data")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    if not a.evaluate and Path(a.out).exists():
        p.error(f"{a.out} exists; choose another --out")
    splits = data.load(a.data)
    if a.evaluate:
        result = evaluate(gpt.load(a.evaluate, a.device), splits, a.device)
    else:
        model, step = train(a, splits)
        torch.save(model.state_dict(), a.out)
        result = dict(selected_step=step, **evaluate(model, splits, a.device))
    print(result)
    Path(a.record).parent.mkdir(parents=True, exist_ok=True)
    Path(a.record).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
