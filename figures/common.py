"""Shared by fig01.py and fig02.py: the random matrices G (N x n, N a multiple of n), the errors, and the plot."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

n = m = 128
S = 100
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "text.latex.preamble": r"\usepackage{amsmath,amssymb,bm}",
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "legend.frameon": False,
})

H = np.array([[1.0]])
while H.shape[0] < n:                                    # Sylvester's Hadamard matrix of order n
    H = np.block([[H, H], [H, -H]])


def gaussian(N):
    return np.random.randn(N, n)


def rademacher(N):
    return np.sign(np.random.randn(N, n))


def hadamard_blocks(N):
    return np.vstack([H * np.sign(np.random.randn(n)) for _ in range(N // n)])


def orthogonal_blocks(N):
    blocks = []
    for _ in range(N // n):
        Q, R = np.linalg.qr(np.random.randn(n, n))
        blocks.append(Q * np.sign(np.diag(R)))               # Haar distributed
    return np.vstack(blocks)


def unit_rows(A):
    return A / np.linalg.norm(A, axis=1, keepdims=True)


def errors(G, W):
    """d_{G,W,x} = ||(2/pi) arcsin(W x) - (1/N) sign(W G^T) sign(G x)||^2, (2.1), for S independent uniformly random
    unit vectors x."""
    X = unit_rows(np.random.randn(S, n))
    est = np.sign(X @ G.T) @ np.sign(W @ G.T).T / len(G)
    return (((2 / np.pi) * np.arcsin(np.clip(X @ W.T, -1.0, 1.0)) - est) ** 2).sum(axis=1)


def plot(sizes, lines, points, ylabel, name, ylim=None):
    """Reference curves (values, style, label) and markers (errors per N, style, label) with standard-deviation bars."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for values, style, label in lines:
        ax.plot(sizes, values, style, linewidth=1, label=label)
    for errs, style, label in points:
        ax.errorbar(sizes, [e.mean() for e in errs], yerr=[e.std() for e in errs], fmt=style, markersize=4, capsize=2,
                    linewidth=0, elinewidth=0.8, label=label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel(r"$N$")
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="0.85", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(Path(__file__).resolve().parent / f"{name}.eps")
