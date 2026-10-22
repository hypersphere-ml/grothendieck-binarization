"""Binarize the baseline's 48 body layers with f_G and N = b n, without fine-tuning, for one construction of G, one b and one
draw of G: the test perplexity (Table 1) and the scaled error on the first 16 validation windows (Table 2).

    python direct.py --construction hadamard --b 1 --seed 1337
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import data
import layers
import model as gpt


@torch.no_grad()
def scaled_error(model, tokens, device, windows=16):
    """For each body layer, (N/m)(1/T) sum_t ||(1/N) sign(W G^T) sign(G x_hat_t) - (2/pi) arcsin(W_hat x_hat_t)||^2 over its
    T = 1024 * windows inputs x_t from the first `windows` windows of tokens; then averaged over the layers, each weighted by
    its number of rows m. Each layer outputs f_G itself here, without the rounding of the surrogate term in
    HypersphereLinear.forward."""
    sse = {}

    def measure(layer, inputs, output):
        x_norm = inputs[0].norm(dim=-1, keepdim=True)
        x_hat = inputs[0] / x_norm.clamp_min(1e-8)
        W_hat = layer.weight / layer.weight.norm(dim=1, keepdim=True).clamp_min(1e-8)
        estimate = layer.estimate(x_hat)
        target = (2 / torch.pi) * torch.asin(F.linear(x_hat, W_hat).clamp(-1.0, 1.0))
        error, count = sse.get(layer, (0.0, 0))
        sse[layer] = error + (estimate - target).double().square().sum().item(), count + target.numel()
        return x_norm * (layer.D * torch.sin(estimate * torch.pi / 2.0)) + (0 if layer.bias is None else layer.bias)

    hooks = [layer.register_forward_hook(measure) for _, layer in gpt.body(model)]
    model.eval()
    for i in range(windows):
        window = torch.from_numpy(tokens[i * data.BLOCK:(i + 1) * data.BLOCK + 1].astype(np.int64)).to(device)
        model(window[None, :-1], window[None, 1:])
    for hook in hooks:
        hook.remove()
    return sum(layer.G.N * error for layer, (error, _) in sse.items()) / sum(count for _, count in sse.values())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--construction", choices=layers.CONSTRUCTIONS, required=True)
    p.add_argument("--b", type=int, required=True)
    p.add_argument("--seed", type=int, required=True, help="the draw of G")
    p.add_argument("--baseline", default="baseline.pt")
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="runs/direct")
    p.add_argument("--device", default="cuda")
    a = p.parse_args()

    splits = data.load(a.data)
    model = gpt.load(a.baseline, a.device)
    layers.binarize(model, a.construction, a.b, a.seed)
    result = dict(construction=a.construction, b=a.b, seed=a.seed,
                  test=data.perplexity(model, splits["test"], a.device),
                  scaled_error=scaled_error(model, splits["validation"], a.device))
    print(result)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / f"{a.construction}_b{a.b}_seed{a.seed}.json").write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
