"""2D yx scatter at a fixed z-plane (in-plane slice through depth)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from viz_mp.db import SimulationDB
from viz_mp.species import get_element, inward_species, outward_species, product_species


def _lam_um(cfg, n: int) -> float:
    return float(getattr(cfg, "SIZE", 0.0)) / max(float(n), 1.0) * 1e6


def plot_2d_yx_slice(
    db: SimulationDB,
    iteration: int,
    z_slice: int,
    *,
    plot_separate: bool = False,
    max_points_per_series: int = 200_000,
    seed: int = 1,
) -> None:
    cfg = db.cfg
    n = db.n_cells
    z_slice = int(np.clip(z_slice, 0, n - 1))
    lam = _lam_um(cfg, n)
    rng = np.random.default_rng(seed)

    series = []

    def add(el: str, color: str, label: str):
        if not el or not db.has_table(el, iteration):
            return
        xyz = db.load_xyz(el, iteration)
        if xyz.size == 0:
            return
        m = xyz[:, 0] == z_slice
        if not np.any(m):
            return
        sl = xyz[m]
        if sl.shape[0] > max_points_per_series:
            sl = sl[rng.choice(sl.shape[0], max_points_per_series, replace=False)]
        y = sl[:, 1].astype(np.float64) * lam
        x = sl[:, 2].astype(np.float64) * lam
        series.append((label, y, x, color))

    inward_colors = ("b", "deeppink", "navy", "purple")
    if getattr(cfg, "INWARD_DIFFUSION", False):
        for i, ox in enumerate(inward_species(cfg)):
            el = str(get_element(ox) or "")
            add(el, inward_colors[i % len(inward_colors)], f"{el} (inward)" if el else f"inward_{i}")

    outward_colors = ("g", "darkorange", "olivedrab", "goldenrod")
    if getattr(cfg, "OUTWARD_DIFFUSION", False):
        for i, ac in enumerate(outward_species(cfg)):
            el = str(get_element(ac) or "")
            add(el, outward_colors[i % len(outward_colors)], f"{el} (outward)" if el else f"outward_{i}")

    if getattr(cfg, "COMPUTE_PRECIPITATION", False):
        product_colors = (
            "crimson",
            "royalblue",
            "seagreen",
            "goldenrod",
            "darkviolet",
            "teal",
            "sienna",
            "deeppink",
        )
        for i, p in enumerate(product_species(cfg)):
            el = str(get_element(p) or "")
            add(el, product_colors[i % len(product_colors)], el if el else f"product_{i}")

    if not series:
        print("viz_mp.plot2d: no points on this z slice for available tables.")
        return

    lim = lam * n

    def _draw_one(label: str, y: np.ndarray, x: np.ndarray, color: str):
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(
            y,
            x,
            s=22,
            c=color,
            label=label,
            marker="s",
            alpha=0.95,
            edgecolors="black",
            linewidths=0.35,
        )
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("y [µm]")
        ax.set_ylabel("x [µm]")
        ax.set_title(f"{label} · yx @ z={z_slice} · iteration {iteration}")
        ax.legend(loc="best", fontsize=8, markerscale=2)
        ax.grid(True, alpha=0.25)
        plt.tight_layout()

    if plot_separate:
        for label, y, x, color in series:
            _draw_one(label, y, x, color)
        plt.show()
        plt.close("all")
        return

    fig, ax = plt.subplots(figsize=(8, 7))
    for label, y, x, color in series:
        ax.scatter(
            y,
            x,
            s=22,
            c=color,
            label=label,
            marker="s",
            alpha=0.95,
            edgecolors="black",
            linewidths=0.35,
        )
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("y [µm]")
    ax.set_ylabel("x [µm]")
    ax.set_title(f"2D yx @ z={z_slice} · iteration {iteration}")
    ax.legend(loc="best", fontsize=8, markerscale=2)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.show()
    plt.close(fig)
