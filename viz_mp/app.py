"""Tk launcher for the new results viewer (does not modify ``results.py``)."""
from __future__ import annotations

import matplotlib

matplotlib.use("TkAgg")

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from viz_mp.db import SimulationDB
from viz_mp.plot2d import plot_2d_yx_slice
from viz_mp.plot3d import plot_3d_combined
from viz_mp.plot_concentration import plot_concentration


class VizMpApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CA 3D MP — Results (viz_mp)")
        self.root.minsize(560, 440)
        self._db: SimulationDB | None = None
        self._db_path = tk.StringVar(value="")
        self._build_ui()

    def _build_ui(self):
        f_file = ttk.LabelFrame(self.root, text="Database", padding=6)
        f_file.pack(fill=tk.X, padx=6, pady=4)
        row = ttk.Frame(f_file)
        row.pack(fill=tk.X)
        ttk.Entry(row, textvariable=self._db_path, width=62).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Button(row, text="Browse…", command=self._browse).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="Load", command=self._load).pack(side=tk.LEFT, padx=2)
        self._status = ttk.Label(f_file, text="No database loaded.", foreground="gray")
        self._status.pack(anchor=tk.W, pady=(4, 0))

        nb = ttk.Notebook(self.root)
        nb.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        t3 = ttk.Frame(nb, padding=8)
        t2 = ttk.Frame(nb, padding=8)
        tc = ttk.Frame(nb, padding=8)
        nb.add(t3, text="3D")
        nb.add(t2, text="2D yx")
        nb.add(tc, text="Concentration")

        self._plot3d_iter = tk.IntVar(value=0)
        self._plot3d_sep = tk.BooleanVar(value=False)
        self._plot2d_iter = tk.IntVar(value=0)
        self._plot2d_z = tk.IntVar(value=0)
        self._plot2d_sep = tk.BooleanVar(value=False)
        self._conc_iter = tk.IntVar(value=0)
        self._conc_mode = tk.StringVar(value="atomic")
        self._conc_cells_mode = tk.StringVar(value="rows")
        self._conc_sep = tk.BooleanVar(value=False)

        lf3 = ttk.LabelFrame(t3, text="3D scatter (µm)", padding=6)
        lf3.pack(fill=tk.X, pady=4)
        r3 = ttk.Frame(lf3)
        r3.pack(fill=tk.X)
        ttk.Label(r3, text="Iteration:").pack(side=tk.LEFT)
        self._lb3 = ttk.Label(r3, text="0")
        self._sc3 = ttk.Scale(
            r3,
            from_=0,
            to=100,
            variable=self._plot3d_iter,
            orient=tk.HORIZONTAL,
            command=self._mk_iter_cb(self._plot3d_iter, self._lb3),
        )
        self._sc3.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self._lb3.pack(side=tk.LEFT)
        ttk.Checkbutton(
            lf3,
            text="Separate windows (one per element/product)",
            variable=self._plot3d_sep,
        ).pack(anchor=tk.W, pady=(4, 0))
        ttk.Button(lf3, text="Plot 3D", command=self._run_3d).pack(anchor=tk.W, pady=(6, 0))

        lf2 = ttk.LabelFrame(t2, text="yx plane at fixed z index", padding=6)
        lf2.pack(fill=tk.X, pady=4)
        r2a = ttk.Frame(lf2)
        r2a.pack(fill=tk.X)
        ttk.Label(r2a, text="Iteration:").pack(side=tk.LEFT)
        self._lb2i = ttk.Label(r2a, text="0")
        self._sc2i = ttk.Scale(
            r2a,
            from_=0,
            to=100,
            variable=self._plot2d_iter,
            orient=tk.HORIZONTAL,
            command=self._mk_iter_cb(self._plot2d_iter, self._lb2i),
        )
        self._sc2i.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self._lb2i.pack(side=tk.LEFT)
        r2b = ttk.Frame(lf2)
        r2b.pack(fill=tk.X, pady=4)
        ttk.Label(r2b, text="z index:").pack(side=tk.LEFT)
        self._lb2z = ttk.Label(r2b, text="0")
        self._sc2z = ttk.Scale(
            r2b,
            from_=0,
            to=100,
            variable=self._plot2d_z,
            orient=tk.HORIZONTAL,
            command=self._mk_iter_cb(self._plot2d_z, self._lb2z),
        )
        self._sc2z.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self._lb2z.pack(side=tk.LEFT)
        ttk.Checkbutton(
            lf2,
            text="Separate windows (one per element/product)",
            variable=self._plot2d_sep,
        ).pack(anchor=tk.W, pady=(2, 0))
        ttk.Button(lf2, text="Plot 2D yx", command=self._run_2d).pack(anchor=tk.W, pady=(6, 0))

        lfc = ttk.LabelFrame(tc, text="Along x (yz-integrated counts / moles)", padding=6)
        lfc.pack(fill=tk.X, pady=4)
        rci = ttk.Frame(lfc)
        rci.pack(fill=tk.X)
        ttk.Label(rci, text="Iteration:").pack(side=tk.LEFT)
        self._lbci = ttk.Label(rci, text="0")
        self._scci = ttk.Scale(
            rci,
            from_=0,
            to=100,
            variable=self._conc_iter,
            orient=tk.HORIZONTAL,
            command=self._mk_iter_cb(self._conc_iter, self._lbci),
        )
        self._scci.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self._lbci.pack(side=tk.LEFT)
        rcm = ttk.Frame(lfc)
        rcm.pack(fill=tk.X, pady=4)
        ttk.Label(rcm, text="Mode:").pack(side=tk.LEFT)
        ttk.Combobox(rcm, textvariable=self._conc_mode, values=("atomic", "mass"), width=10, state="readonly").pack(side=tk.LEFT, padx=4)
        ttk.Label(rcm, text="Cells count:").pack(side=tk.LEFT, padx=(12, 0))
        ttk.Combobox(rcm, textvariable=self._conc_cells_mode, values=("rows", "unique"), width=8, state="readonly").pack(side=tk.LEFT, padx=4)
        ttk.Checkbutton(
            lfc,
            text="Separate windows (one per element/product)",
            variable=self._conc_sep,
        ).pack(anchor=tk.W, pady=(2, 0))
        ttk.Button(lfc, text="Plot concentration", command=self._run_conc).pack(anchor=tk.W, pady=(6, 0))

    @staticmethod
    def _mk_iter_cb(var, lbl):
        def _cb(v):
            lbl.config(text=str(int(float(v))))

        return _cb

    def _browse(self):
        p = filedialog.askopenfilename(
            title="Select simulation database",
            filetypes=[("SQLite", "*.db *.sqlite *.sqlite3"), ("All files", "*.*")],
        )
        if p:
            self._db_path.set(p)

    def _load(self):
        path = self._db_path.get().strip()
        if not path:
            messagebox.showwarning("No file", "Select a database file.")
            return
        self._status.config(text="Loading…", foreground="gray")
        self.root.update_idletasks()
        try:
            db = SimulationDB(path)
        except Exception as e:
            self._db = None
            self._status.config(text=f"Error: {e}", foreground="red")
            messagebox.showerror("Load failed", str(e))
            return
        self._db = db
        last = int(getattr(db, "last_i", 0) or 0)
        n = int(db.n_cells or 1)
        self._status.config(
            text=f"Loaded: {path}  (last_i={last}, N={n})",
            foreground="green",
        )
        for sc, var, lb in (
            (self._sc3, self._plot3d_iter, self._lb3),
            (self._sc2i, self._plot2d_iter, self._lb2i),
            (self._scci, self._conc_iter, self._lbci),
        ):
            var.set(min(var.get(), last))
            sc.config(to=max(last, 0))
            lb.config(text=str(var.get()))
        self._plot2d_z.set(min(self._plot2d_z.get(), n - 1))
        self._sc2z.config(to=max(0, n - 1))
        self._plot2d_z.set(min(self._plot2d_z.get(), n - 1))
        self._lb2z.config(text=str(self._plot2d_z.get()))

    def _need_db(self) -> bool:
        if self._db is None:
            messagebox.showwarning("No database", "Load a database first.")
            return True
        return False

    def _run_viz(self, fn):
        try:
            fn()
        except Exception as e:
            messagebox.showerror("Plot error", str(e))

    def _run_3d(self):
        if self._need_db():
            return
        it = int(self._plot3d_iter.get())
        self._run_viz(lambda: plot_3d_combined(self._db, it, plot_separate=bool(self._plot3d_sep.get())))

    def _run_2d(self):
        if self._need_db():
            return
        it = int(self._plot2d_iter.get())
        zs = int(self._plot2d_z.get())
        self._run_viz(
            lambda: plot_2d_yx_slice(self._db, it, zs, plot_separate=bool(self._plot2d_sep.get()))
        )

    def _run_conc(self):
        if self._need_db():
            return
        it = int(self._conc_iter.get())
        mode = self._conc_mode.get() or "atomic"
        cm = self._conc_cells_mode.get() or "rows"
        if mode not in ("atomic", "mass"):
            mode = "atomic"
        if cm not in ("rows", "unique"):
            cm = "rows"
        self._run_viz(
            lambda: plot_concentration(
                self._db,
                it,
                mode,
                plot_separate=bool(self._conc_sep.get()),
                cell_count_mode=cm,
            )
        )

    def run(self):
        self.root.mainloop()


def main():
    VizMpApp().run()
