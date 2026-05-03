"""SQLite access: pickled config, iteration tables ``{name}_iter_{k}`` with columns (z, y, x)."""
import pickle
import sqlite3 as sql
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from viz_mp import config_view
from viz_mp.species import get_element, product_species


def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


class SimulationDB:
    def __init__(self, path: str):
        self.path = path
        self.conn = sql.connect(path)
        self.c = self.conn.cursor()
        self.cfg = self._load_config_required()
        self.n_cells = int(getattr(self.cfg, "N_CELLS_PER_AXIS", 0) or 0)
        self.last_i, self.elapsed_s = self._load_time_parameters()
        self._all_iter_tables: Set[str] = self._fetch_iter_table_names()
        self._available_prefixes = self._scan_iter_prefixes_from_tables(self._all_iter_tables)

    def close(self):
        self.conn.close()

    def _load_config_required(self):
        self.c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='PickledConfig'"
        )
        if self.c.fetchone() is None:
            raise ValueError(
                "This database has no PickledConfig table. "
                "The new viewer only supports the current simulation format."
            )
        self.c.execute("SELECT pickled_data FROM PickledConfig")
        row = self.c.fetchone()
        if not row:
            raise ValueError("PickledConfig is empty.")
        raw = pickle.loads(row[0])
        return config_view.apply_pickled_config(raw)

    def _load_time_parameters(self) -> Tuple[int, float]:
        try:
            self.c.execute("SELECT last_i, elapsed_time FROM time_parameters")
            r = self.c.fetchone()
            if r:
                return int(r[0] or 0), float(r[1] or 0.0)
        except sql.Error:
            pass
        return 0, 0.0

    def _fetch_iter_table_names(self) -> Set[str]:
        self.c.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE '%_iter_%'"
        )
        return {r[0] for r in self.c.fetchall()}

    @staticmethod
    def _scan_iter_prefixes_from_tables(all_names: Set[str]) -> Set[str]:
        out: Set[str] = set()
        for name in all_names:
            if "_iter_" not in name:
                continue
            out.add(name.rsplit("_iter_", 1)[0])
        return out

    def available_iterations(self) -> List[int]:
        iters: Set[int] = set()
        for name in self._all_iter_tables:
            if "_iter_" not in name:
                continue
            tail = name.rsplit("_iter_", 1)[1]
            try:
                iters.add(int(tail))
            except ValueError:
                continue
        return sorted(iters)

    def has_table(self, species_table_prefix: str, iteration: int) -> bool:
        return f"{species_table_prefix}_iter_{iteration}" in self._all_iter_tables

    def load_xyz(self, table_prefix: str, iteration: int) -> np.ndarray:
        """
        Return (N, 3) int array [z, y, x]. Empty array if table missing or no rows.
        """
        tname = f"{table_prefix}_iter_{iteration}"
        if tname not in self._all_iter_tables:
            return np.zeros((0, 3), dtype=np.int32)
        q = f"SELECT z, y, x FROM {_quote_ident(tname)}"
        try:
            self.c.execute(q)
            rows = self.c.fetchall()
        except sql.Error:
            return np.zeros((0, 3), dtype=np.int32)
        if not rows:
            return np.zeros((0, 3), dtype=np.int32)
        return np.asarray(rows, dtype=np.int32)

    def resolve_table_prefix(self, logical_or_element: str) -> Optional[str]:
        """
        If ``logical_or_element`` is already a table prefix for some iteration, return it.
        Else map legacy logical names to element strings from config (same idea as old viz).
        """
        if logical_or_element in self._available_prefixes:
            return logical_or_element
        aliases = self._build_prefix_aliases()
        mapped = aliases.get(logical_or_element, logical_or_element)
        if mapped in self._available_prefixes:
            return mapped
        return None

    def _build_prefix_aliases(self) -> Dict[str, str]:
        cfg = self.cfg
        oxidants = getattr(cfg, "OXIDANTS", []) or []
        actives = getattr(cfg, "ACTIVES", []) or []
        aliases: Dict[str, str] = {}
        if isinstance(oxidants, list):
            for idx, ox in enumerate(oxidants):
                el = get_element(ox)
                if el:
                    aliases[f"inward_{idx}"] = el
        if isinstance(actives, list):
            for idx, ac in enumerate(actives):
                el = get_element(ac)
                if el:
                    aliases[f"outward_{idx}"] = el
        for idx, p in enumerate(product_species(cfg)):
            el = get_element(p)
            if el:
                aliases[f"product_{idx}"] = el
                key = None
                if isinstance(p, dict):
                    key = p.get("key")
                else:
                    key = getattr(p, "KEY", None) or getattr(p, "key", None)
                if key:
                    aliases[str(key)] = el
        return aliases

    def element_table_if_exists(self, element: str, iteration: int) -> Optional[str]:
        """Return table base name (same as element) if ``{element}_iter_{it}`` exists."""
        t = f"{element}_iter_{iteration}"
        if t in self._all_iter_tables:
            return element
        return None
