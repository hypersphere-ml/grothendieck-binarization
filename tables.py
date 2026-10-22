"""Tables 1-3, and the extra longer-budget comparison, from a directory of run results laid out as results/ is.

    python tables.py            # the runs reported in the paper
    python tables.py runs       # your runs
"""
import json
import statistics
import sys
from pathlib import Path

BITS = (1, 2, 3, 6)
CONSTRUCTIONS = ("hadamard", "orthogonal", "gaussian", "rademacher")
METHODS = ("ours", "scalar", "rotated")
WIDTH = 30


def summary(values, expected):
    """Mean and sample standard deviation; "-" for a cell without runs, [k/n] for a cell with fewer runs than the paper."""
    if not values:
        return "-"
    mean = f"{statistics.mean(values):#.5g}".rstrip(".")
    text = mean if len(values) == 1 else f"{mean} +- {statistics.stdev(values):.3g}"
    return text if len(values) == expected else f"{text} [{len(values)}/{expected}]"


def choose_lr(ladder):
    """The learning rate selected from the seed-1337 runs at each rate: the best perplexity on validation partition B, or the
    lowest rate when no rate improves on the starting weights. Also whether the rate is inside the rates tried."""
    ladder = sorted(ladder, key=lambda r: r["lr"])
    if not ladder:
        return None, True
    if all(r["selected_step"] == 0 for r in ladder):
        return ladder[0]["lr"], True
    best = min(ladder, key=lambda r: (r["val_b"], r["lr"]))
    return best["lr"], ladder[0]["lr"] < best["lr"] < ladder[-1]["lr"]


def main(directory):
    root = Path(directory)
    if not (root / "baseline.json").exists():
        sys.exit(f"{root / 'baseline.json'} is missing: run python train_baseline.py --evaluate baseline.pt first")
    baseline = json.loads((root / "baseline.json").read_text())["test"]
    direct = [json.loads(p.read_text()) for p in sorted((root / "direct").glob("*.json"))]
    runs = [json.loads(p.read_text()) for p in sorted((root / "finetune").glob("*.json"))]

    def ratio(r):
        return r["test"] / baseline

    def direct_runs(construction, b):
        return [r for r in direct if r["construction"] == construction and r["b"] == b]

    def finetuned(method, b, **match):
        return [r for r in runs if r["method"] == method and r["b"] == b and all(r[k] == v for k, v in match.items())]

    print(f"Baseline test perplexity {baseline:.4f}\n")
    print("Table 1: test perplexity / baseline, over draws of G")
    print("b  " + "".join(f"{c:>{WIDTH}}" for c in CONSTRUCTIONS))
    for b in BITS:
        print(f"{b}  " + "".join(f"{summary([ratio(r) for r in direct_runs(c, b)], 5):>{WIDTH}}" for c in CONSTRUCTIONS))

    print("\nTable 2: scaled error, over draws of G (1 - 4/pi^2 = 0.5947)")
    print("b  " + "".join(f"{c:>{WIDTH}}" for c in CONSTRUCTIONS[:2]))
    for b in BITS:
        cells = [summary([r["scaled_error"] for r in direct_runs(c, b)], 5) for c in CONSTRUCTIONS[:2]]
        print(f"{b}  " + "".join(f"{c:>{WIDTH}}" for c in cells))

    print("\nTable 3: test perplexity / baseline")
    print("without fine-tuning" + "".join(f"{m:>{WIDTH}}" for m in METHODS))
    for b in BITS:
        cells = [summary([ratio(r) for r in direct_runs("hadamard", b)], 5),
                 summary([ratio(r) for r in finetuned("scalar", b, steps=0)], 1),
                 summary([ratio(r) for r in finetuned("rotated", b, steps=0)], 5)]
        print(f"{b:<19}" + "".join(f"{c:>{WIDTH}}" for c in cells))
    print("with fine-tuning (learning rate)")
    for b in BITS:
        line = f"{b:<19}"
        for m in METHODS:
            lr, inside = choose_lr(finetuned(m, b, steps=1600, seed=1337))
            cell = summary([ratio(r) for r in finetuned(m, b, steps=1600, lr=lr)], 3)
            if lr is not None:
                cell += f" ({lr:g})" + ("" if inside else "!")
            line += f"{cell:>{WIDTH}}"
        print(line)
    print("(! = the chosen rate is at an end of the rates in this directory; to repeat the selection, extend the grid)")

    print("\nExtra, not in the paper: one bit, seed 1337, at the chosen rate, 6,400 against 1,600 steps")
    for m in METHODS:
        lr = choose_lr(finetuned(m, 1, steps=1600, seed=1337))[0]
        long, short = (finetuned(m, 1, steps=s, seed=1337, lr=lr) for s in (6400, 1600))
        if long and short:
            print(f"  {m:8}{ratio(long[0]):.2f} (from {ratio(short[0]):.2f})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
