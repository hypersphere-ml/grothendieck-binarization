"""Figure 3: the sample estimate of E_G[d_{G,I}] for the Gaussian G of Theorem 2.8, the orthogonal block G of Theorem 2.9
and the Hadamard block G of Corollary 2.11: S draws of G, S of x for each, with W = I, n = m = 128 and S = 100. The solid
line is the lower bound of Theorem 2.10 for G with entries +-1, without its O(1/n) term."""
import numpy as np

from common import S, errors, gaussian, hadamard_blocks, m, n, orthogonal_blocks, plot

np.random.seed(0)
sizes = n * 2 ** np.arange(8)
W = np.eye(n)


def curve(draw):
    return [np.concatenate([errors(draw(N), W) for _ in range(S)]) for N in sizes]


gaussian_errors = curve(gaussian)
for N in sizes:                    # the published figure also drew Rademacher matrices and their inputs here, unplotted;
    for _ in range(S):             # drawing the same numbers reproduces its orthogonal and Hadamard block points
        np.random.randn(N + S, n)
points = [(gaussian_errors, "co", "Gaussian"), (curve(orthogonal_blocks), "gD", "Orthogonal block"),
          (curve(hadamard_blocks), "r^", "Hadamard block")]
c = 1 - 4 / np.pi**2
plot(sizes, [(m / sizes, "k:", r"$m/N$"), (c * m / sizes, "k--", r"$(1-\tfrac{4}{\pi^2})\,m/N$"),
             (0.026 + 0.363 * n / sizes, "k-", r"$0.026 + 0.363\,n/N$")],
     points, r"$\overline{d}_{\boldsymbol G, \boldsymbol I}$", "fig02", ylim=(2.0**-9, 1.4))
