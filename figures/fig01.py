"""Figure 2: the sample estimate (2.7) of E_G[d_G] for the Rademacher G of Corollary 2.4 and the Hadamard block G of
Corollary 2.5: S draws of G, S of W for each, S of x for each W, with n = m = 128 and S = 100."""
import numpy as np

from common import S, errors, hadamard_blocks, m, n, plot, rademacher, unit_rows

np.random.seed(0)
sizes = n * 2 ** np.arange(7)


def all_errors(draw, N):
    out = []
    for _ in range(S):
        G = draw(N)
        for _ in range(S):
            out.append(errors(G, unit_rows(np.random.randn(m, n))))
    return np.concatenate(out)


rad = [all_errors(rademacher, N) for N in sizes]
had = [all_errors(hadamard_blocks, N) for N in sizes]
c = 1 - 4 / np.pi**2
plot(sizes, [(m / sizes, "k-", r"$m/N$"), (c * m / sizes, "k--", r"$(1-\tfrac{4}{\pi^2})\,m/N$")],
     [(rad, "bs", "Rademacher"), (had, "r^", "Hadamard block")], r"$\overline{d}_{\boldsymbol{G}}$", "fig01")
