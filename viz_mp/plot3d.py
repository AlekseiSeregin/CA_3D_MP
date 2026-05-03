"""Matplotlib 3D scatter from iteration tables (optional subsampling for speed)."""
from __future__ import annotations

from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from viz_mp.db import SimulationDB
from viz_mp.species import get_element, inward_species, outward_species, product_species


def _lam_um(cfg, n: int) -> float:
    return float(getattr(cfg, "SIZE", 0.0)) / max(float(n), 1.0) * 1e6


def _subsample(xyz: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    n = xyz.shape[0]
    if n <= max_points:
        return xyz
    idx = rng.choice(n, size=max_points, replace=False)
    return xyz[idx]


def plot_3d_combined(
    db: SimulationDB,
    iteration: int,
    *,
    plot_separate: bool = False,
    max_points_per_series: int = 250_000,
    seed: int = 0,
) -> None:
    """3D scatter at one iteration (combined or one window per species)."""
    cfg = db.cfg
    n = db.n_cells
    lam = _lam_um(cfg, n)
    rng = np.random.default_rng(seed)

    series: List[Tuple[str, np.ndarray, str]] = []

    inward_colors = ("b", "deeppink", "navy", "purple")
    if getattr(cfg, "INWARD_DIFFUSION", False):
        for i, ox in enumerate(inward_species(cfg)):
            el = get_element(ox)
            if not el or not db.has_table(str(el), iteration):
                continue
            xyz = db.load_xyz(str(el), iteration)
            if xyz.size:
                color = inward_colors[i % len(inward_colors)]
                series.append((f"{el} (inward)", _subsample(xyz, max_points_per_series, rng), color))

    outward_colors = ("g", "darkorange", "olivedrab", "goldenrod")
    if getattr(cfg, "OUTWARD_DIFFUSION", False):
        for i, ac in enumerate(outward_species(cfg)):
            el = get_element(ac)
            if not el or not db.has_table(str(el), iteration):
                continue
            xyz = db.load_xyz(str(el), iteration)
            if xyz.size:
                color = outward_colors[i % len(outward_colors)]
                series.append((f"{el} (outward)", _subsample(xyz, max_points_per_series, rng), color))

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
            el = get_element(p)
            if not el or not db.has_table(str(el), iteration):
                continue
            xyz = db.load_xyz(str(el), iteration)
            if xyz.size:
                color = product_colors[i % len(product_colors)]
                series.append((str(el), _subsample(xyz, max_points_per_series, rng), color))

    if not series:
        print("viz_mp.plot3d: no particle tables for this iteration.")
        return

    lim = lam * n

    def _draw_one(name: str, xyz: np.ndarray, c: str):
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection="3d")
        xs = xyz[:, 2].astype(np.float64) * lam
        ys = xyz[:, 1].astype(np.float64) * lam
        zs = xyz[:, 0].astype(np.float64) * lam
        ax.scatter(
            xs,
            ys,
            zs,
            marker="s",
            s=18.0,
            c=c,
            label=name,
            depthshade=False,
            alpha=0.95,
            edgecolors="black",
            linewidths=0.3,
        )
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_zlim(0, lim)
        ax.set_xlabel("x [µm]")
        ax.set_ylabel("y [µm]")
        ax.set_zlabel("z [µm]")
        ax.set_title(f"{name} · iteration {iteration}")
        ax.legend(loc="upper right", fontsize=8, markerscale=2)
        plt.tight_layout()

    if plot_separate:
        for name, xyz, c in series:
            _draw_one(name, xyz, c)
        plt.show()
        plt.close("all")
        return

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for name, xyz, c in series:
        xs = xyz[:, 2].astype(np.float64) * lam
        ys = xyz[:, 1].astype(np.float64) * lam
        zs = xyz[:, 0].astype(np.float64) * lam
        ax.scatter(
            xs,
            ys,
            zs,
            marker="s",
            s=18.0,
            c=c,
            label=name,
            depthshade=False,
            alpha=0.95,
            edgecolors="black",
            linewidths=0.3,
        )
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_zlim(0, lim)
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    ax.set_zlabel("z [µm]")
    ax.set_title(f"3D · iteration {iteration}")
    ax.legend(loc="upper right", fontsize=8, markerscale=2)
    plt.tight_layout()
    plt.show()
    plt.close(fig)
