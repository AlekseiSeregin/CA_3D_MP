"""
Per-x (depth) profiles: counts come from yz planes (binning on x in stored rows z,y,x).

* ``cells`` — count in plane / N² (rows or unique (y,z), see ``cell_count_mode``).
* ``atomic`` — mole % in plane: each species moles / sum(all moles in plane including matrix).
* ``mass`` — wt% from mole fractions × molar masses / Σ(xⱼ Mⱼ).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Literal, Tuple

import numpy as np

from viz_mp import planes
from viz_mp.db import SimulationDB
from viz_mp.species import get_element, get_field, inward_species, outward_species, product_species

CellCountMode = Literal["rows", "unique"]
ConcMode = Literal["cells", "atomic", "mass"]


def _getf(item: Any, key: str, default: float = 0.0) -> float:
    v = get_field(item, key, default)
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _species_counts(db: SimulationDB, iteration: int, n: int, count_fn, species: List[Any]) -> List[Tuple[Any, str, np.ndarray]]:
    out: List[Tuple[Any, str, np.ndarray]] = []
    for idx, sp in enumerate(species):
        el = get_element(sp)
        label = el or f"species_{idx}"
        if not el or not db.has_table(el, iteration):
            out.append((sp, label, np.zeros(n, dtype=np.float64)))
            continue
        xyz = db.load_xyz(el, iteration)
        out.append((sp, label, count_fn(xyz, n).astype(np.float64)))
    return out


def _matrix_eq_sink_moles(
    n: int,
    outward_counts: List[Tuple[Any, str, np.ndarray]],
    product_counts: List[Tuple[Any, str, np.ndarray]],
) -> np.ndarray:
    """Per-plane matrix-equivalent sink moles for role-based outward/product species."""
    z = np.zeros(n, dtype=np.float64)

    outward_eq: dict[str, float] = {}
    for sp, el, counts in outward_counts:
        eq = _getf(sp, "eq_matrix_moles_per_cell")
        outward_eq[el] = eq
        z += counts * eq

    for p, _el, counts in product_counts:
        thr_out = _getf(p, "threshold_outward")
        outward_el = get_field(p, "outward_element", None)
        eq = outward_eq.get(str(outward_el), 0.0) if outward_el else 0.0
        if eq > 0.0 and thr_out > 0.0:
            z += counts * eq * thr_out
        else:
            # Fallback for products that do not consume outward species.
            z += counts * _getf(p, "moles_per_cell")

    return z


@dataclass
class ConcentrationResult:
    depth_m: np.ndarray
    labels: List[str]
    values: np.ndarray
    ylabel: str
    title: str


def compute_concentration_result(
    db: SimulationDB,
    iteration: int,
    mode: ConcMode,
    cell_count_mode: CellCountMode = "rows",
) -> ConcentrationResult:
    cfg = db.cfg
    n = int(db.n_cells)
    if n <= 0:
        raise ValueError("N_CELLS_PER_AXIS is missing or invalid.")

    count_fn = planes.counts_per_depth_x if cell_count_mode == "rows" else planes.unique_cells_per_depth_x

    n_plane = float(n * n)
    depth = np.linspace(0.0, float(getattr(cfg, "SIZE", 0.0) or 0.0), n, dtype=np.float64)

    oxid = _species_counts(
        db, iteration, n, count_fn, inward_species(cfg) if getattr(cfg, "INWARD_DIFFUSION", False) else []
    )
    actv = _species_counts(
        db, iteration, n, count_fn, outward_species(cfg) if getattr(cfg, "OUTWARD_DIFFUSION", False) else []
    )
    prod = _species_counts(
        db, iteration, n, count_fn, product_species(cfg) if getattr(cfg, "COMPUTE_PRECIPITATION", False) else []
    )

    sink = _matrix_eq_sink_moles(n, actv, prod)
    matrix_moles = n_plane * _getf(cfg.MATRIX, "MOLES_PER_CELL") - sink

    labels: List[str] = []
    series: List[np.ndarray] = []

    if mode == "cells":
        for _sp, el, c in oxid:
            labels.append(f"{el} (inward)")
            series.append(100.0 * c / n_plane)
        for _sp, el, c in actv:
            labels.append(f"{el} (outward)")
            series.append(100.0 * c / n_plane)
        for _sp, el, c in prod:
            labels.append(str(el))
            series.append(100.0 * c / n_plane)
        if not series:
            labels.append("(no species tables)")
            series.append(np.zeros(n))
        ylabel = "Plane occupancy [% of N² yz sites]"
        title = f"Cells · iteration {iteration} · count={cell_count_mode}"
        return ConcentrationResult(depth, labels, np.vstack(series), ylabel, title)

    # moles per plane for atomic / mass
    mol_o: List[Tuple[str, np.ndarray]] = []
    for ox, el, c in oxid:
        mpc = _getf(ox, "MOLES_PER_CELL")
        mol_o.append((el, c * mpc))

    mol_a: List[Tuple[str, np.ndarray]] = []
    for ac, el, c in actv:
        mpc = _getf(ac, "MOLES_PER_CELL")
        mol_a.append((el, c * mpc))

    mol_p: List[Tuple[Any, str, np.ndarray]] = []
    for p, el, c in prod:
        mpc = _getf(p, "MOLES_PER_CELL")
        mol_p.append((p, el, c * mpc))

    inward_m = np.zeros(n, dtype=np.float64)
    for _, m in mol_o:
        inward_m += m
    outward_m = np.zeros(n, dtype=np.float64)
    for _, m in mol_a:
        outward_m += m
    prod_m = np.zeros(n, dtype=np.float64)
    for _, _, m in mol_p:
        prod_m += m

    whole_moles = matrix_moles + inward_m + outward_m + prod_m
    whole_moles = np.maximum(whole_moles, 1e-300)

    if mode == "atomic":
        labels.append(f"matrix ({getattr(cfg.MATRIX, 'ELEMENT', 'matrix')})")
        series.append(matrix_moles * 100.0 / whole_moles)
        for el, m in mol_o:
            labels.append(el)
            series.append(m * 100.0 / whole_moles)
        for el, m in mol_a:
            labels.append(el)
            series.append(m * 100.0 / whole_moles)
        for _p, el, m in mol_p:
            labels.append(el)
            series.append(m * 100.0 / whole_moles)
        ylabel = "Mole-based % in yz plane"
        title = f"Atomic (mole %) · iteration {iteration}"
        return ConcentrationResult(depth, labels, np.vstack(series), ylabel, title)

    # mass from mole fractions × M
    m_mat = _getf(cfg.MATRIX, "MOLAR_MASS", 58.6934)
    xs = [matrix_moles / whole_moles]
    ms = [m_mat]
    labels = [f"matrix ({getattr(cfg.MATRIX, 'ELEMENT', 'matrix')})"]

    for ox, (el, m) in zip((row[0] for row in oxid), mol_o):
        M = _getf(ox, "MOLAR_MASS", 0.0) or 0.0
        xs.append(m / whole_moles)
        ms.append(M)
        labels.append(el)

    for ac, (el, m) in zip((row[0] for row in actv), mol_a):
        M = _getf(ac, "MOLAR_MASS", 0.0) or 0.0
        xs.append(m / whole_moles)
        ms.append(M)
        labels.append(el)

    for p, el, m in mol_p:
        M = _getf(p, "MOLAR_MASS", 0.0) or 0.0
        xs.append(m / whole_moles)
        ms.append(M)
        labels.append(el)

    x_arr = np.vstack(xs)
    m_arr = np.asarray(ms, dtype=np.float64)
    w_num = x_arr * m_arr[:, None]
    w_den = np.sum(w_num, axis=0)
    w_den = np.maximum(w_den, 1e-300)
    wt = w_num * 100.0 / w_den
    ylabel = "Mass % (from mole fractions × M)"
    title = f"Mass % · iteration {iteration}"
    return ConcentrationResult(depth, labels, wt, ylabel, title)
