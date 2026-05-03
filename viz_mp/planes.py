"""Per-depth (x) reductions: counts in each yz plane from (z, y, x) coordinate rows."""
import numpy as np


def counts_per_depth_x(xyz: np.ndarray, n_cells: int) -> np.ndarray:
    """
    ``xyz`` rows are (z, y, x) as stored in SQLite. Bin by x (depth index).

    Returns length ``n_cells`` vector: number of stored rows per x index.
    """
    if xyz is None or xyz.size == 0:
        return np.zeros(n_cells, dtype=np.int64)
    x = np.asarray(xyz[:, 2], dtype=np.int64)
    x = x[(x >= 0) & (x < n_cells)]
    if x.size == 0:
        return np.zeros(n_cells, dtype=np.int64)
    return np.bincount(x, minlength=n_cells).astype(np.int64, copy=False)


def unique_cells_per_depth_x(xyz: np.ndarray, n_cells: int) -> np.ndarray:
    """Count distinct (y, z) sites per x (optional stricter 'cells' semantics)."""
    if xyz is None or xyz.size == 0:
        return np.zeros(n_cells, dtype=np.int64)
    z = np.asarray(xyz[:, 0], dtype=np.int64)
    y = np.asarray(xyz[:, 1], dtype=np.int64)
    x = np.asarray(xyz[:, 2], dtype=np.int64)
    m = (x >= 0) & (x < n_cells) & (y >= 0) & (y < n_cells) & (z >= 0) & (z < n_cells)
    if not np.any(m):
        return np.zeros(n_cells, dtype=np.int64)
    z, y, x = z[m], y[m], x[m]
    keys = x.astype(np.int64) * (n_cells * n_cells) + y.astype(np.int64) * n_cells + z.astype(np.int64)
    order = np.argsort(keys, kind="mergesort")
    keys_s = keys[order]
    first = np.ones(len(keys_s), dtype=bool)
    first[1:] = keys_s[1:] != keys_s[:-1]
    ux = x[order][first]
    cnt = np.bincount(ux, minlength=n_cells)
    return cnt.astype(np.int64, copy=False)
