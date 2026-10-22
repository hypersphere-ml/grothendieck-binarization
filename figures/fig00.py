"""Figure 1: the maps u -> sign(g_1^T u) and u -> sign(G u) on the upper hemisphere of S^2, for a 6 x 3 Gaussian G."""
from pathlib import Path
from colorsys import hls_to_rgb
from itertools import combinations, product
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

# Fixed Gaussian matrix used throughout the figure.
N = 6
G = np.random.default_rng(7).standard_normal((N, 3))
OUT = Path(__file__).resolve().parent

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 12,
    "savefig.facecolor": "white",
    "path.simplify": False,
})


def sphere_axes(fig, rect):
    ax = fig.add_axes(rect)
    ax.set(xlim=(-1.08, 1.08), ylim=(-1.08, 1.08), aspect="equal")
    ax.set_axis_off()
    return ax


def spherical_cell(normals):
    """Vertices of {u in S^2 : normals @ u >= 0}, in boundary order."""
    A = normals / np.linalg.norm(normals, axis=1)[:, None]
    vertices = []
    for i, j in combinations(range(len(A)), 2):
        v = np.cross(A[i], A[j])
        if np.linalg.norm(v) < 1e-10:
            continue
        v /= np.linalg.norm(v)
        for p in (v, -v):
            if np.all(A @ p >= -1e-9):
                if not any(np.linalg.norm(p - q) < 1e-8 for q in vertices):
                    vertices.append(p.copy())
    if len(vertices) < 3:
        return None
    V = np.array(vertices)
    center = V.sum(axis=0)
    center /= np.linalg.norm(center)
    e1 = V[0] - np.dot(V[0], center) * center
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(center, e1)
    return V[np.argsort(np.arctan2(V @ e2, V @ e1))]


def projected_boundary(vertices):
    """Sample each spherical edge and project to the (u1,u2)-plane."""
    arcs = []
    for a, b in zip(vertices, np.roll(vertices, -1, axis=0)):
        angle = np.arccos(np.clip(a @ b, -1, 1))
        t = np.linspace(0, 1, max(8, int(np.ceil(angle / 0.006))), endpoint=False)
        arc = (1 - t[:, None]) * a + t[:, None] * b
        arc /= np.linalg.norm(arc, axis=1)[:, None]
        arcs.append(arc[:, :2])
    return np.vstack(arcs)


# Visible sign cells on the hemisphere u3 >= 0.
cells = {}
for signs in product((-1, 1), repeat=N):
    normals = np.vstack((np.array(signs)[:, None] * G, [0.0, 0.0, 1.0]))
    vertices = spherical_cell(normals)
    if vertices is not None:
        cells[signs] = projected_boundary(vertices)


# One reproducible color per realizable sign code.
all_codes = sorted(set(cells) | {tuple(-np.array(s)) for s in cells})
palette = np.array([
    hls_to_rgb(i / len(all_codes), 0.73 + 0.10 * (i % 2), 0.50)
    for i in range(len(all_codes))
])
np.random.default_rng(19).shuffle(palette)
colors = dict(zip(all_codes, palette))


def draw_partition(ax):
    for signs, boundary in cells.items():
        ax.add_patch(
            Polygon(boundary, facecolor=colors[signs], edgecolor="0.22",
                    linewidth=0.45, joinstyle="round")
        )
    ax.add_patch(Circle((0, 0), 1, fill=False, edgecolor="0.15", linewidth=0.8))


def draw_one_row(ax):
    """Partition induced by sign(g_1^T u)."""
    g = G[0] / np.linalg.norm(G[0])
    a = np.array([g[1], -g[0], 0.0])
    a /= np.linalg.norm(a)
    b = np.cross(g, a)
    if b[2] < 0:
        b = -b
    t = np.linspace(0, np.pi, 600)[:, None]
    boundary = np.cos(t) * a + np.sin(t) * b

    for sign, color in [(+1, "#9fcddd"), (-1, "#edd7b0")]:
        e = np.cross([0, 0, 1], a)
        if sign * (g @ e) < 0:
            e = -e
        rim = -np.cos(t) * a + np.sin(t) * e
        points = np.vstack((boundary, rim))[:, :2]
        ax.add_patch(Polygon(points, facecolor=color, edgecolor="0.22", linewidth=0.65))

    ax.add_patch(Circle((0, 0), 1, fill=False, edgecolor="0.15", linewidth=0.8))
    ax.text(0, 0.85, r"$+1$", ha="center", va="center", fontsize=20)
    ax.text(0, -0.18, r"$-1$", ha="center", va="center", fontsize=20)


def save_panel(draw_function, name):
    fig = plt.figure(figsize=(3.35, 3.35))
    ax = sphere_axes(fig, [0.02, 0.02, 0.96, 0.96])
    draw_function(ax)
    fig.savefig(OUT / f"{name}.eps", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


if __name__ == "__main__":
    save_panel(draw_one_row, "fig00a")
    save_panel(draw_partition, "fig00b")
