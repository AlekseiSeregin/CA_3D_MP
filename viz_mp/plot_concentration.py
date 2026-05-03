"""Matplotlib line plots for plane-resolved concentrations."""
from __future__ import annotations

import matplotlib.pyplot as plt

from viz_mp.concentrations import CellCountMode, ConcMode, compute_concentration_result
from viz_mp.db import SimulationDB


def plot_concentration(
    db: SimulationDB,
    iteration: int,
    mode: ConcMode,
    *,
    plot_separate: bool = False,
    cell_count_mode: CellCountMode = "rows",
) -> None:
    res = compute_concentration_result(db, iteration, mode, cell_count_mode=cell_count_mode)
    depth_um = res.depth_m * 1e6

    def _style(ax, title: str):
        ax.set_xlabel("Depth x [µm]")
        ax.set_ylabel(res.ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)

    if plot_separate:
        for i, lab in enumerate(res.labels):
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(depth_um, res.values[i], label=lab, linewidth=1.0)
            _style(ax, f"{lab} · {res.title}")
            plt.tight_layout()
        plt.show()
        plt.close("all")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, lab in enumerate(res.labels):
        ax.plot(depth_um, res.values[i], label=lab, linewidth=1.0)
    _style(ax, res.title)
    plt.tight_layout()
    plt.show()
    plt.close(fig)
