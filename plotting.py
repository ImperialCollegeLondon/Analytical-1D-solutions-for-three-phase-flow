# Author: Zhi Zheng
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from baker_model import baker_krg, baker_kro, baker_krw
from data_and_corey import SGC, SORG, SWC


def _ensure_dir(save):
    directory = os.path.dirname(os.path.abspath(save))
    if directory:
        os.makedirs(directory, exist_ok=True)


def ternary_coords(Sw, So, Sg):
    Sw = np.asarray(Sw, dtype=float)
    Sg = np.asarray(Sg, dtype=float)
    return Sw + 0.5 * Sg, np.sqrt(3.0) * Sg / 2.0


def draw_ternary_frame(ax):
    h = np.sqrt(3.0) / 2.0
    ax.plot([0, 1, 0.5, 0], [0, 0, h, 0], color="black", lw=1.5)
    for i in np.arange(0.1, 1.0, 0.1):
        lines = [
            (ternary_coords(1 - i, 0, i), ternary_coords(0, 1 - i, i)),
            (ternary_coords(i, 1 - i, 0), ternary_coords(i, 0, 1 - i)),
            (ternary_coords(0, i, 1 - i), ternary_coords(1 - i, i, 0)),
        ]
        for (x1, y1), (x2, y2) in lines:
            ax.plot([x1, x2], [y1, y2], color="0.86", lw=0.35, zorder=0)
    ax.text(1.04, -0.045, r"$S_w$", fontsize=12)
    ax.text(-0.06, -0.045, r"$S_o$", fontsize=12)
    ax.text(0.48, h + 0.04, r"$S_g$", fontsize=12)
    ax.set_aspect("equal")
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.07, h + 0.08)
    ax.axis("off")


def _mobile_triangle(ax):
    Sw_b = np.array([SWC, 1.0 - SORG, SWC, SWC])
    Sg_b = np.array([0.0, 0.0, 1.0 - SWC - SORG, 0.0])
    So_b = 1.0 - Sw_b - Sg_b
    x, y = ternary_coords(Sw_b, So_b, Sg_b)
    ax.fill(x, y, color="#e8f2ff", alpha=0.8, edgecolor="#7aa6d8", lw=0.8, zorder=0)


def _draw_rarefaction_network(ax, cp, f2, ng=7, nmax=180, h=0.005):
    from analytical_solver import integrate_rarefaction

    sw_values = np.linspace(SWC + 0.05, 1.0 - SORG - 0.05, ng)
    sg_values = np.linspace(0.03, 1.0 - SWC - SORG - 0.05, ng)
    styles = {
        1: {"color": "#1f77b4", "alpha": 0.22, "lw": 0.55},
        2: {"color": "#d62728", "alpha": 0.20, "lw": 0.55},
    }
    for Sw0 in sw_values:
        for Sg0 in sg_values:
            if 1.0 - Sw0 - Sg0 < SORG + 0.015:
                continue
            for fam in (1, 2):
                for direction in (1, -1):
                    try:
                        path = integrate_rarefaction(
                            Sw0, Sg0, fam, direction, cp, f2, Nmax=nmax, h=h
                        )
                    except Exception:
                        continue
                    path = np.asarray(path, dtype=float)
                    if path.ndim != 2 or path.shape[0] < 3 or path.shape[1] < 3:
                        continue
                    finite = np.isfinite(path[:, 0]) & np.isfinite(path[:, 1]) & np.isfinite(path[:, 2])
                    path = path[finite]
                    if len(path) < 3:
                        continue
                    x, y = ternary_coords(path[:, 0], path[:, 1], path[:, 2])
                    ax.plot(x, y, zorder=1, **styles[fam])


def _solution_segments(sol):
    segments = []
    for key in ("slow_path", "fast_path"):
        arr = sol.get(key)
        if arr is None:
            continue
        arr = np.asarray(arr, dtype=float)
        if arr.ndim == 2 and arr.shape[0] > 0 and arr.shape[1] >= 3:
            segments.append((key, arr[:, :4] if arr.shape[1] >= 4 else arr[:, :3]))
    if not segments and all(k in sol for k in ("Sw", "So", "Sg", "xi")):
        arr = np.column_stack([sol["Sw"], sol["So"], sol["Sg"], sol["xi"]])
        segments.append(("profile", arr))
    return segments


def _plot_arrow(ax, x, y, color):
    if len(x) < 4:
        return
    i = len(x) // 2
    ax.annotate(
        "",
        xy=(x[min(i + 1, len(x) - 1)], y[min(i + 1, len(y) - 1)]),
        xytext=(x[i], y[i]),
        arrowprops={"arrowstyle": "->", "lw": 1.5, "color": color},
        zorder=5,
    )


def plot_three_phase_relative_permeability(cp, f2, save):
    _ensure_dir(save)
    n = 75
    Sw_grid = np.linspace(SWC, 1.0 - SORG, n)
    Sg_grid = np.linspace(SGC, 1.0 - SWC - SORG, n)
    rows = []
    for Sw in Sw_grid:
        for Sg in Sg_grid:
            So = 1.0 - Sw - Sg
            if So >= SORG - 1e-10 and Sg >= SGC - 1e-10:
                rows.append((Sw, So, Sg))
    pts = np.asarray(rows)
    x, y = ternary_coords(pts[:, 0], pts[:, 1], pts[:, 2])
    values = [
        np.array([baker_krw(Sw, Sg, cp, f2) for Sw, _, Sg in pts]),
        np.array([baker_kro(Sw, Sg, cp, f2) for Sw, _, Sg in pts]),
        np.array([baker_krg(Sw, Sg, cp, f2) for Sw, _, Sg in pts]),
    ]
    titles = [r"$k_{rw}$", r"$k_{ro}$", r"$k_{rg}$"]
    cmaps = ["Blues", "Reds", "Greens"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for ax, val, title, cmap in zip(axes, values, titles, cmaps):
        draw_ternary_frame(ax)
        cf = ax.tricontourf(x, y, val, levels=22, cmap=cmap)
        ax.tricontour(x, y, val, levels=8, colors="0.25", linewidths=0.25, alpha=0.45)
        ax.set_title(title, fontsize=13)
        fig.colorbar(cf, ax=ax, shrink=0.78, pad=0.02)
    fig.suptitle("Baker three-phase relative permeability", fontsize=13)
    fig.tight_layout()
    fig.savefig(save, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_analytical_ternary_path(sol, save, cp=None, f2=None):
    _ensure_dir(save)
    fig, ax = plt.subplots(figsize=(7.0, 6.2))
    draw_ternary_frame(ax)
    _mobile_triangle(ax)
    if cp is not None and f2 is not None:
        _draw_rarefaction_network(ax, cp, f2)
    colors = {"slow_path": "#1f77b4", "fast_path": "#d62728", "profile": "#222222"}
    labels = {"slow_path": "slow branch", "fast_path": "fast branch", "profile": "path"}
    ax.plot([], [], color="#1f77b4", lw=0.8, alpha=0.35, label="slow rarefaction grid")
    ax.plot([], [], color="#d62728", lw=0.8, alpha=0.35, label="fast rarefaction grid")
    for key, arr in _solution_segments(sol):
        x, y = ternary_coords(arr[:, 0], arr[:, 1], arr[:, 2])
        color = colors.get(key, "#222222")
        ax.plot(x, y, color=color, lw=2.2, label=labels.get(key, key), zorder=3)
        _plot_arrow(ax, x, y, color)
    for label, state, marker in [("L", sol["L"], "o"), ("R", sol["R"], "s")]:
        x, y = ternary_coords(state[0], state[1], state[2])
        ax.plot(x, y, marker, ms=7, color="black", zorder=6)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(7, 6), fontsize=11)
    ax.legend(loc="upper right", fontsize=9, frameon=False)
    ax.set_title("Analytical ternary path from L to R", fontsize=13)
    fig.savefig(save, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_numerical_ternary(num, sol, save, cp=None, f2=None):
    _ensure_dir(save)
    fig, ax = plt.subplots(figsize=(7.4, 6.4))
    draw_ternary_frame(ax)
    _mobile_triangle(ax)
    if cp is not None and f2 is not None:
        _draw_rarefaction_network(ax, cp, f2, ng=6, nmax=150)

    R = sol["R"]
    dist_r = np.sqrt((num["Sw"] - R[0]) ** 2 + (num["Sg"] - R[2]) ** 2)
    active = dist_r > 0.003
    if not np.any(active):
        active = np.ones_like(num["Sw"], dtype=bool)
    x, y = ternary_coords(num["Sw"][active], num["So"][active], num["Sg"][active])
    sc = ax.scatter(x, y, c=num["xi"][active], s=8, cmap="viridis", edgecolors="none", zorder=4)
    fig.colorbar(sc, ax=ax, shrink=0.72, label=r"$\xi=x/t$")

    for _, arr in _solution_segments(sol):
        xa, ya = ternary_coords(arr[:, 0], arr[:, 1], arr[:, 2])
        ax.plot(xa, ya, color="black", ls="--", lw=1.4, alpha=0.75, zorder=5, label="analytical path")

    for label, state, marker in [("L", sol["L"], "o"), ("R", sol["R"], "s")]:
        xs, ys = ternary_coords(state[0], state[1], state[2])
        ax.plot(xs, ys, marker, ms=7, color="black", zorder=6)
        ax.annotate(label, (xs, ys), textcoords="offset points", xytext=(7, 6), fontsize=11)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_title("Numerical ternary path with analytical overlay", fontsize=13)
    fig.savefig(save, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _profile_arrays(sol):
    xi = np.asarray(sol["xi"], dtype=float)
    fields = [np.asarray(sol[k], dtype=float) for k in ("Sw", "So", "Sg")]
    order = np.argsort(xi, kind="mergesort")
    return xi[order], [f[order] for f in fields]


def plot_num_vs_analytical_logxi(num, sol, save):
    _ensure_dir(save)
    xi_a, ana = _profile_arrays(sol)
    xi_n = np.asarray(num["xi"], dtype=float)
    num_fields = [np.asarray(num[k], dtype=float) for k in ("Sw", "So", "Sg")]
    names = [r"$S_w$", r"$S_o$", r"$S_g$"]
    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    positive = np.concatenate([xi_a[np.isfinite(xi_a) & (xi_a > 0)], xi_n[np.isfinite(xi_n) & (xi_n > 0)]])
    xi_min = max(float(np.min(positive)) * 0.85, 1e-8)
    xi_max = float(np.max(positive)) * 1.05

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.0))
    axes = axes.ravel()
    for k in range(3):
        ax = axes[k]
        ma = np.isfinite(xi_a) & (xi_a > 0) & np.isfinite(ana[k])
        mn = np.isfinite(xi_n) & (xi_n > 0) & np.isfinite(num_fields[k])
        ax.semilogx(xi_a[ma], ana[k][ma], color=colors[k], lw=2.0, label="analytical")
        ax.semilogx(xi_n[mn], num_fields[k][mn], color=colors[k], ls="--", lw=1.3, label="numerical")
        ax.set_xlim(xi_min, xi_max)
        ax.set_ylim(-0.03, 1.03)
        ax.set_title(names[k])
        ax.set_xlabel(r"$\xi=x/t$")
        ax.grid(True, which="both", alpha=0.28)
        ax.legend(fontsize=8)

    ax = axes[3]
    for k in range(3):
        ma = np.isfinite(xi_a) & (xi_a > 0) & np.isfinite(ana[k])
        mn = np.isfinite(xi_n) & (xi_n > 0) & np.isfinite(num_fields[k])
        ax.semilogx(xi_a[ma], ana[k][ma], color=colors[k], lw=2.0, label=f"{names[k]} analytical")
        ax.semilogx(xi_n[mn], num_fields[k][mn], color=colors[k], ls="--", lw=1.2, alpha=0.85)
    ax.set_xlim(xi_min, xi_max)
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Overlay")
    ax.set_xlabel(r"$\xi=x/t$")
    ax.grid(True, which="both", alpha=0.28)
    ax.legend(fontsize=8, ncol=1)

    fig.suptitle("Numerical vs analytical saturation profiles", fontsize=13)
    fig.tight_layout()
    fig.savefig(save, dpi=180, bbox_inches="tight")
    plt.close(fig)
