"""The random matrices G of Section 2, our hypersphere binarized layer f_G of Section 3.1, and the scalar quantization of
Section 3.5, each applied to the 48 linear layers of the transformer blocks (model.body, the "body layers")."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import body


# The constructions of G, an N x n matrix with N = b n. Each is a module that maps t to t G^T.

def fwht(x):
    """Unnormalized Walsh-Hadamard transform (Sylvester's matrix) along the last dimension, whose length is a power of 2."""
    *prefix, n = x.shape
    h = 1
    while h < n:
        a, b = x.view(*prefix, n // (2 * h), 2, h).unbind(dim=-2)
        x = torch.stack([a + b, a - b], dim=-2).reshape(*prefix, n)
        h *= 2
    return x


def paley12():
    """Paley's Hadamard matrix of order 12."""
    residues = {i * i % 11 for i in range(1, 11)}
    H = torch.ones(12, 12)
    for i in range(11):
        for j in range(11):
            H[i + 1, j + 1] = -1 if i == j or (i - j) % 11 not in residues else 1
    return H


class HadamardBlocks(nn.Module):
    """G = [H S_1; ...; H S_b], S_k diagonal with independent random signs. H is Sylvester's Hadamard matrix when n is a
    power of 2, and otherwise the Kronecker product of Paley's matrix of order 12 and Sylvester's of order n/12; so n
    must be a power of 2 or 12 times one."""

    def __init__(self, n, b, device):
        super().__init__()
        self.N = b * n
        self.paley = (n & (n - 1)) != 0
        self.register_buffer("H12", paley12().to(device) if self.paley else None)
        self.register_buffer("signs", torch.sign(torch.randn(b, n, device=device)))

    def forward(self, t):
        x = t.unsqueeze(-2) * self.signs
        if not self.paley:
            return fwht(x).flatten(-2)
        x = fwht(x.unflatten(-1, (12, -1)))
        return (x.transpose(-1, -2) @ self.H12.T).transpose(-1, -2).flatten(-3)


class OrthogonalBlocks(nn.Module):
    """G = [Q_1; ...; Q_b] with independent Haar-random orthogonal n x n blocks Q_k."""

    def __init__(self, n, b, device):
        super().__init__()
        self.N = b * n
        Q, R = torch.linalg.qr(torch.randn(b, n, n, device=device))
        self.register_buffer("Q", Q * torch.sign(torch.diagonal(R, dim1=-2, dim2=-1)).unsqueeze(-2))  # Haar, not just QR

    def forward(self, t):
        return torch.einsum("...m,knm->...kn", t, self.Q).flatten(-2)


class Dense(nn.Module):
    def __init__(self, G):
        super().__init__()
        self.N = G.shape[0]
        self.register_buffer("G", G)

    def forward(self, t):
        return t @ self.G.T


CONSTRUCTIONS = {
    "hadamard": HadamardBlocks,
    "orthogonal": OrthogonalBlocks,
    "gaussian": lambda n, b, device: Dense(torch.randn(b * n, n, device=device)),
    "rademacher": lambda n, b, device: Dense(torch.sign(torch.randn(b * n, n, device=device))),
}


class HypersphereLinear(nn.Module):
    """f_G(x; W) = ||x|| D sin(pi/(2N) sign(W G^T) sign(G x)) + bias, with D fixed at the row norms D_0 of the weights it is
    built from. In training, its gradient is that of D_0 W_hat x + bias, where W_hat has the normalized rows of the
    current weights."""

    def __init__(self, linear, G):
        super().__init__()
        self.weight, self.bias, self.G = linear.weight, linear.bias, G
        self.register_buffer("D", linear.weight.detach().norm(dim=1))

    def estimate(self, x_hat):
        """(1/N) sign(W G^T) sign(G x_hat), which approximates (2/pi) arcsin(W_hat x_hat)."""
        return torch.sign(self.G(x_hat)) @ torch.sign(self.G(self.weight)).T / self.G.N

    def forward(self, x):
        x_norm = x.norm(dim=-1, keepdim=True)
        bias = 0 if self.bias is None else self.bias
        with torch.no_grad():
            y = x_norm * (self.D * torch.sin(self.estimate(x / x_norm.clamp_min(1e-8)) * torch.pi / 2.0)) + bias
        W_hat = self.weight / self.weight.norm(dim=1, keepdim=True).clamp_min(1e-8)
        surrogate = self.D * F.linear(x, W_hat) + bias
        # The value y and the gradient of the surrogate. (y + surrogate) - surrogate is not exactly y in floating point; the
        # reported runs include that rounding, also at evaluation, and it changes perplexities, so the expression is kept.
        return y.to(surrogate.dtype) + surrogate - surrogate.detach()


def binarize(model, construction, b, seed):
    """Replace each layer of model.body (the 48 linear layers of this repository's GPT) by f_G with N = b n; the G are drawn
    in block order after torch.manual_seed(seed)."""
    torch.manual_seed(seed)
    for name, linear in body(model):
        G = CONSTRUCTIONS[construction](linear.in_features, b, linear.weight.device)
        model.set_submodule(name, HypersphereLinear(linear, G))


# Scalar quantization. The spacing s is the least-squares one among s = f max|v| / 2^(b-1), f from these candidates.
WEIGHT_FRACTIONS = torch.linspace(0.1, 1.0, 64).tolist()     # per row of W
INPUT_FRACTIONS = torch.linspace(0.3, 1.0, 8).tolist()       # per input vector


@torch.no_grad()
def round_scalar(v, bits, fractions):
    """Round each row of v to the nearest of the levels +-s(k - 1/2), k = 1, ..., 2^(b-1), with the spacing s that
    minimizes the squared error: s = 2 mean|v| at one bit, otherwise the best candidate. Returns the rounded v and s."""
    v = v.float()
    if bits == 1:
        s = 2.0 * v.abs().mean(dim=-1, keepdim=True).clamp_min(1e-9)
        return torch.sign(v) * (s / 2), s
    half = 1 << (bits - 1)
    vmax = v.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9)
    best = None
    for f in fractions:
        s = vmax * f / half
        q = ((v / s).floor().clamp(-half, half - 1) + 0.5) * s
        err = (v - q).pow(2).sum(dim=-1, keepdim=True)
        if best is None:
            best = q, s, err
        else:
            take = err < best[2]
            best = torch.where(take, q, best[0]), torch.where(take, s, best[1]), torch.where(take, err, best[2])
    return best[0], best[1]


class ClippedIdentity(torch.autograd.Function):
    """Value q; gradient that of the identity in x where `inside`, zero elsewhere."""

    @staticmethod
    def forward(ctx, x, q, inside):
        ctx.save_for_backward(inside)
        return q

    @staticmethod
    def backward(ctx, grad):
        inside, = ctx.saved_tensors
        return grad * inside.to(grad.dtype), None, None


class ScalarLinear(nn.Module):
    """W x + bias ~ [W R^T] [R x] + bias, the b-bit rounding [.] applied to each row of W R^T and to R x, with
    R = H S / sqrt(n) for rotated scalar quantization and R = I otherwise. Straight-through gradients: the identity
    through the rounding of the weights, and through the rounding of the input except at coordinates z with
    |z| >= 2^(b-1) s."""

    def __init__(self, linear, bits, rotation=None):
        super().__init__()
        self.weight, self.bias, self.bits, self.rotation = linear.weight, linear.bias, bits, rotation
        self.cache = None      # the rounded weights, recomputed only when the weights change

    def rotate(self, t):
        if self.rotation is None:
            return t
        with torch.autocast(t.device.type, enabled=False):
            return self.rotation(t.float()) * (1.0 / math.sqrt(t.shape[-1]))

    def forward(self, x):
        z = self.rotate(x)
        z_rounded, s = round_scalar(z, self.bits, INPUT_FRACTIONS)
        z = ClippedIdentity.apply(z, z_rounded.to(z.dtype), z.float().abs() < s * (1 << (self.bits - 1)))
        W = self.rotate(self.weight)
        key = self.weight._version, self.weight.device, self.weight.dtype
        if self.cache is None or self.cache[0] != key:
            with torch.no_grad(), torch.autocast(W.device.type, enabled=False):
                self.cache = key, round_scalar(W, self.bits, WEIGHT_FRACTIONS)[0]
        return F.linear(z, W + (self.cache[1] - W).detach(), self.bias)


def quantize(model, bits, rotation_seed=None):
    """Replace each layer of model.body by b-bit scalar quantization, rotated when rotation_seed is given: then each
    layer's S is drawn, in block order after torch.manual_seed(rotation_seed), as the signs of a one-block Hadamard G."""
    if rotation_seed is not None:
        torch.manual_seed(rotation_seed)
    for name, linear in body(model):
        R = HadamardBlocks(linear.in_features, 1, linear.weight.device) if rotation_seed is not None else None
        model.set_submodule(name, ScalarLinear(linear, bits, R))
