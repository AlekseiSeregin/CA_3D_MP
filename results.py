"""
Results visualisation launcher with GUI.
Load a simulation database and run visualisations via buttons and sliders.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from visualisation import Visualisation, plot_kinetics, plot_kinetics_mult_comb


def _parse_layers(s: str):
    """Parse '1, 3, 4' or '1-5' into a tuple of ints."""
    s = (s or "").strip()
    if not s:
        return ()
    out = []
    for part in s.replace(" ", "").split(","):
        if "-" in part:
            a, b = part.split("-", 1)
            out.extend(range(int(a.strip()), int(b.strip()) + 1))
        else:
            out.append(int(part))
    return tuple(out)


class VisualisationApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("CA 3D MP – Results Visualisation")
        self.root.minsize(520, 420)
        self._visualise = None
        self._db_path = tk.StringVar(value="")
        self._build_ui()

    def _build_ui(self):
        # --- File section ---
        f_file = ttk.LabelFrame(self.root, text="Database", padding=6)
        f_file.pack(fill=tk.X, padx=6, pady=4)

        row = ttk.Frame(f_file)
        row.pack(fill=tk.X)
        ttk.Entry(row, textvariable=self._db_path, width=60).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))
        ttk.Button(row, text="Browse…", command=self._browse_db).pack(side=tk.LEFT, padx=2)
        ttk.Button(row, text="Load", command=self._load_db).pack(side=tk.LEFT, padx=2)

        self._status = ttk.Label(f_file, text="No database loaded.", foreground="gray")
        self._status.pack(anchor=tk.W, pady=(4, 0))

        # --- Tabs ---
        self._notebook = ttk.Notebook(self.root)
        self._notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=4)

        self._tab_3d = ttk.Frame(self._notebook, padding=8)
        self._tab_2d = ttk.Frame(self._notebook, padding=8)
        self._tab_conc = ttk.Frame(self._notebook, padding=8)
        self._tab_kinetics = ttk.Frame(self._notebook, padding=8)
        self._tab_other = ttk.Frame(self._notebook, padding=8)

        self._notebook.add(self._tab_3d, text="3D")
        self._notebook.add(self._tab_2d, text="2D")
        self._notebook.add(self._tab_conc, text="Concentration")
        self._notebook.add(self._tab_kinetics, text="Kinetics")
        self._notebook.add(self._tab_other, text="Other")

        self._build_tab_3d()
        self._build_tab_2d()
        self._build_tab_conc()
        self._build_tab_kinetics()
        self._build_tab_other()

    def _build_tab_3d(self):
        g = self._tab_3d
        # Animate 3D
        lf_anim = ttk.LabelFrame(g, text="Animate 3D", padding=6)
        lf_anim.pack(fill=tk.X, pady=4)
        self._anim_separate = tk.BooleanVar(value=False)
        self._anim_const_cam = tk.BooleanVar(value=False)
        ttk.Checkbutton(lf_anim, text="Separate windows (one per quantity)", variable=self._anim_separate).pack(anchor=tk.W)
        ttk.Checkbutton(lf_anim, text="Fixed camera", variable=self._anim_const_cam).pack(anchor=tk.W)
        ttk.Button(lf_anim, text="Run animation", command=self._run_animate_3d).pack(anchor=tk.W, pady=(6, 0))

        # Plot 3D
        lf_plot = ttk.LabelFrame(g, text="Plot 3D (single frame)", padding=6)
        lf_plot.pack(fill=tk.X, pady=4)
        self._plot3d_separate = tk.BooleanVar(value=False)
        self._plot3d_const_cam = tk.BooleanVar(value=False)
        self._plot3d_iter = tk.IntVar(value=0)
        ttk.Checkbutton(lf_plot, text="Separate windows (one per quantity)", variable=self._plot3d_separate).pack(anchor=tk.W)
        ttk.Checkbutton(lf_plot, text="Fixed camera", variable=self._plot3d_const_cam).pack(anchor=tk.W)
        row_i = ttk.Frame(lf_plot)
        row_i.pack(fill=tk.X, pady=2)
        ttk.Label(row_i, text="Iteration:").pack(side=tk.LEFT, padx=(0, 4))
        self._scale_plot3d = ttk.Scale(
            row_i, from_=0, to=100, variable=self._plot3d_iter, orient=tk.HORIZONTAL, length=200,
            command=lambda v: self._lbl_plot3d_iter.config(text=str(int(float(v))))
        )
        self._scale_plot3d.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self._lbl_plot3d_iter = ttk.Label(row_i, text="0")
        self._lbl_plot3d_iter.pack(side=tk.LEFT)
        ttk.Button(lf_plot, text="Plot 3D", command=self._run_plot_3d).pack(anchor=tk.W, pady=(6, 0))

    def _build_tab_2d(self):
        g = self._tab_2d
        lf = ttk.LabelFrame(g, text="Plot 2D", padding=6)
        lf.pack(fill=tk.X, pady=4)
        self._plot2d_separate = tk.BooleanVar(value=False)
        self._plot2d_iter = tk.IntVar(value=0)
        self._plot2d_use_default_slice = tk.BooleanVar(value=True)
        self._plot2d_slice = tk.IntVar(value=0)
        ttk.Checkbutton(lf, text="Separate windows (one per quantity)", variable=self._plot2d_separate).pack(anchor=tk.W)
        row_i = ttk.Frame(lf)
        row_i.pack(fill=tk.X, pady=2)
        ttk.Label(row_i, text="Iteration:").pack(side=tk.LEFT, padx=(0, 4))
        self._scale_plot2d_iter = ttk.Scale(
            row_i, from_=0, to=100, variable=self._plot2d_iter, orient=tk.HORIZONTAL, length=200,
            command=lambda v: self._lbl_plot2d_iter.config(text=str(int(float(v))))
        )
        self._scale_plot2d_iter.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self._lbl_plot2d_iter = ttk.Label(row_i, text="0")
        self._lbl_plot2d_iter.pack(side=tk.LEFT)
        ttk.Checkbutton(lf, text="Use default slice (middle)", variable=self._plot2d_use_default_slice).pack(anchor=tk.W)
        row_s = ttk.Frame(lf)
        row_s.pack(fill=tk.X, pady=2)
        ttk.Label(row_s, text="Slice (z):").pack(side=tk.LEFT, padx=(0, 4))
        self._scale_plot2d_slice = ttk.Scale(
            row_s, from_=0, to=100, variable=self._plot2d_slice, orient=tk.HORIZONTAL, length=200,
            command=lambda v: self._lbl_plot2d_slice.config(text=str(int(float(v))))
        )
        self._scale_plot2d_slice.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self._lbl_plot2d_slice = ttk.Label(row_s, text="0")
        self._lbl_plot2d_slice.pack(side=tk.LEFT)
        ttk.Button(lf, text="Plot 2D", command=self._run_plot_2d).pack(anchor=tk.W, pady=(6, 0))

    def _build_tab_conc(self):
        g = self._tab_conc
        # Plot concentration
        lf_plot = ttk.LabelFrame(g, text="Plot concentration", padding=6)
        lf_plot.pack(fill=tk.X, pady=4)
        self._conc_plot_separate = tk.BooleanVar(value=False)
        self._conc_plot_iter = tk.IntVar(value=0)
        self._conc_type = tk.StringVar(value="cells")
        self._conc_analytic = tk.BooleanVar(value=False)
        ttk.Checkbutton(lf_plot, text="Separate windows (oxidants | actives & products)", variable=self._conc_plot_separate).pack(anchor=tk.W)
        row_i = ttk.Frame(lf_plot)
        row_i.pack(fill=tk.X, pady=2)
        ttk.Label(row_i, text="Iteration:").pack(side=tk.LEFT, padx=(0, 4))
        self._scale_conc_iter = ttk.Scale(
            row_i, from_=0, to=100, variable=self._conc_plot_iter, orient=tk.HORIZONTAL, length=200,
            command=lambda v: self._lbl_conc_iter.config(text=str(int(float(v))))
        )
        self._scale_conc_iter.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self._lbl_conc_iter = ttk.Label(row_i, text="0")
        self._lbl_conc_iter.pack(side=tk.LEFT)
        row_c = ttk.Frame(lf_plot)
        row_c.pack(fill=tk.X, pady=2)
        ttk.Label(row_c, text="Type:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Combobox(row_c, textvariable=self._conc_type, values=("cells", "atomic"), width=10, state="readonly").pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(lf_plot, text="Analytic solution", variable=self._conc_analytic).pack(anchor=tk.W)
        ttk.Button(lf_plot, text="Plot concentration", command=self._run_plot_concentration).pack(anchor=tk.W, pady=(6, 0))

        # Animate concentration
        lf_anim = ttk.LabelFrame(g, text="Animate concentration", padding=6)
        lf_anim.pack(fill=tk.X, pady=4)
        self._conc_anim_type = tk.StringVar(value="cells")
        self._conc_anim_analytic = tk.BooleanVar(value=False)
        row_ca = ttk.Frame(lf_anim)
        row_ca.pack(fill=tk.X, pady=2)
        ttk.Label(row_ca, text="Type:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Combobox(row_ca, textvariable=self._conc_anim_type, values=("cells", "atomic"), width=10, state="readonly").pack(side=tk.LEFT, padx=2)
        ttk.Checkbutton(lf_anim, text="Analytic solution", variable=self._conc_anim_analytic).pack(anchor=tk.W)
        ttk.Button(lf_anim, text="Run concentration animation", command=self._run_animate_concentration).pack(anchor=tk.W, pady=(6, 0))

    def _build_tab_kinetics(self):
        g = self._tab_kinetics
        lf = ttk.LabelFrame(g, text="Plot kinetics (from CSV)", padding=6)
        lf.pack(fill=tk.X, pady=4)
        ttk.Label(lf, text="Layers to plot (e.g. 1,3,4,6 or 1-10):").pack(anchor=tk.W)
        self._kinetics_layers = tk.StringVar(value="1, 3, 4, 6, 14, 20, 27")
        ttk.Entry(lf, textvariable=self._kinetics_layers, width=40).pack(fill=tk.X, pady=2)
        self._kinetics_with_kinetic = tk.BooleanVar(value=False)
        ttk.Checkbutton(lf, text="Include kinetic curves", variable=self._kinetics_with_kinetic).pack(anchor=tk.W)
        ttk.Button(lf, text="Plot kinetics (choose CSV file…)", command=self._run_plot_kinetics).pack(anchor=tk.W, pady=(6, 0))

        lf2 = ttk.LabelFrame(g, text="Plot kinetics (multiple DBs)", padding=6)
        lf2.pack(fill=tk.X, pady=4)
        self._kinetics_num_dbs = tk.IntVar(value=2)
        row_n = ttk.Frame(lf2)
        row_n.pack(fill=tk.X, pady=2)
        ttk.Label(row_n, text="Number of CSV files:").pack(side=tk.LEFT, padx=(0, 4))
        ttk.Spinbox(row_n, from_=1, to=20, textvariable=self._kinetics_num_dbs, width=5).pack(side=tk.LEFT, padx=2)
        ttk.Button(lf2, text="Plot kinetics (choose CSV files…)", command=self._run_plot_kinetics_mult).pack(anchor=tk.W, pady=(6, 0))

    def _build_tab_other(self):
        g = self._tab_other
        lf = ttk.LabelFrame(g, text="Phase size", padding=6)
        lf.pack(fill=tk.X, pady=4)
        self._phase_iter = tk.IntVar(value=0)
        row_i = ttk.Frame(lf)
        row_i.pack(fill=tk.X, pady=2)
        ttk.Label(row_i, text="Iteration:").pack(side=tk.LEFT, padx=(0, 4))
        self._scale_phase_iter = ttk.Scale(
            row_i, from_=0, to=100, variable=self._phase_iter, orient=tk.HORIZONTAL, length=200,
            command=lambda v: self._lbl_phase_iter.config(text=str(int(float(v))))
        )
        self._scale_phase_iter.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        self._lbl_phase_iter = ttk.Label(row_i, text="0")
        self._lbl_phase_iter.pack(side=tk.LEFT)
        ttk.Button(lf, text="Calculate phase size", command=self._run_calculate_phase_size).pack(anchor=tk.W, pady=(6, 0))

        lf_h = ttk.LabelFrame(g, text="Plot h", padding=6)
        lf_h.pack(fill=tk.X, pady=4)
        ttk.Button(lf_h, text="Plot h", command=self._run_plot_h).pack(anchor=tk.W)

        lf_plane0 = ttk.LabelFrame(g, text="Plane-0 product tracking", padding=6)
        lf_plane0.pack(fill=tk.X, pady=4)
        ttk.Button(
            lf_plane0,
            text="Plot jmatpro vs existing (plane 0)",
            command=self._run_plot_plane0_product_tracking,
        ).pack(anchor=tk.W)

    def _browse_db(self):
        path = filedialog.askopenfilename(
            title="Select simulation database",
            filetypes=[("SQLite", "*.db *.sqlite *.sqlite3"), ("All files", "*.*")]
        )
        if path:
            self._db_path.set(path)

    def _load_db(self):
        path = self._db_path.get().strip()
        if not path:
            messagebox.showwarning("No file", "Please select a database file.")
            return
        self._status.config(text="Loading…", foreground="gray")
        self.root.update_idletasks()
        # Load on main thread so SQLite connection is used only on main thread.
        try:
            vis = Visualisation(path)
            self._on_db_loaded(vis, path, None)
        except Exception as e:
            self._on_db_loaded(None, path, str(e))

    def _on_db_loaded(self, vis, path, error):
        if error:
            self._status.config(text=f"Error: {error}", foreground="red")
            messagebox.showerror("Load failed", error)
            return
        self._visualise = vis
        last_i = getattr(vis, "last_i", 0) or 0
        axlim = getattr(vis, "axlim", 100) or 100
        self._status.config(text=f"Loaded: {path}  (last_i={last_i}, axlim={axlim})", foreground="green")

        # Update slider ranges and labels
        self._plot3d_iter.set(min(self._plot3d_iter.get(), last_i))
        self._scale_plot3d.config(to=last_i)
        self._lbl_plot3d_iter.config(text=str(self._plot3d_iter.get()))

        self._plot2d_iter.set(min(self._plot2d_iter.get(), last_i))
        self._scale_plot2d_iter.config(to=last_i)
        self._plot2d_slice.set(min(self._plot2d_slice.get(), axlim - 1))
        self._scale_plot2d_slice.config(to=max(0, axlim - 1))
        self._lbl_plot2d_iter.config(text=str(self._plot2d_iter.get()))
        self._lbl_plot2d_slice.config(text=str(self._plot2d_slice.get()))

        self._conc_plot_iter.set(min(self._conc_plot_iter.get(), last_i))
        self._scale_conc_iter.config(to=last_i)
        self._lbl_conc_iter.config(text=str(self._conc_plot_iter.get()))

        self._phase_iter.set(min(self._phase_iter.get(), last_i))
        self._scale_phase_iter.config(to=last_i)
        self._lbl_phase_iter.config(text=str(self._phase_iter.get()))

    def _ensure_loaded(self):
        if self._visualise is None:
            messagebox.showwarning("No database", "Load a database first.")
            return True
        return False

    def _run_viz(self, fn):
        """Run a visualisation on the main thread (required for SQLite + Matplotlib)."""
        try:
            fn()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _run_animate_3d(self):
        if self._ensure_loaded():
            return
        v = self._visualise
        sep = self._anim_separate.get()
        cam = self._anim_const_cam.get()
        self._run_viz(lambda: v.animate_3d(animate_separate=sep, const_cam_pos=cam))

    def _run_plot_3d(self):
        if self._ensure_loaded():
            return
        v = self._visualise
        it = self._plot3d_iter.get()
        self._run_viz(lambda: v.plot_3d(
            plot_separate=self._plot3d_separate.get(),
            iteration=it,
            const_cam_pos=self._plot3d_const_cam.get(),
        ))

    def _run_plot_2d(self):
        if self._ensure_loaded():
            return
        v = self._visualise
        it = self._plot2d_iter.get()
        slice_pos = None if self._plot2d_use_default_slice.get() else self._plot2d_slice.get()
        self._run_viz(lambda: v.plot_2d(
            plot_separate=self._plot2d_separate.get(),
            iteration=it,
            slice_pos=slice_pos,
        ))

    def _run_plot_concentration(self):
        if self._ensure_loaded():
            return
        v = self._visualise
        self._run_viz(lambda: v.plot_concentration(
            plot_separate=self._conc_plot_separate.get(),
            iteration=self._conc_plot_iter.get(),
            conc_type=self._conc_type.get() or "cells",
            analytic_sol=self._conc_analytic.get(),
        ))

    def _run_animate_concentration(self):
        if self._ensure_loaded():
            return
        v = self._visualise
        self._run_viz(lambda: v.animate_concentration(
            conc_type=self._conc_anim_type.get() or "cells",
            analytic_sol=self._conc_anim_analytic.get(),
        ))

    def _run_plot_kinetics(self):
        layers = _parse_layers(self._kinetics_layers.get())
        if not layers:
            messagebox.showwarning("Layers", "Enter at least one layer (e.g. 1,3,4,6).")
            return
        path = filedialog.askopenfilename(
            title="Select kinetics CSV",
            filetypes=[("CSV / text", "*.csv *.txt"), ("All files", "*.*")]
        )
        if not path:
            return
        with_kinetic = self._kinetics_with_kinetic.get()
        self._run_viz(lambda: plot_kinetics(layers, with_kinetic=with_kinetic, file_path=path))

    def _run_plot_kinetics_mult(self):
        try:
            n = int(self._kinetics_num_dbs.get())
        except (tk.TclError, ValueError):
            n = 2
        layers = _parse_layers(self._kinetics_layers.get())
        if not layers:
            messagebox.showwarning("Layers", "Enter at least one layer (e.g. 1,3,4,6).")
            return
        self._run_viz(lambda: plot_kinetics_mult_comb(layers, n, with_kinetic=self._kinetics_with_kinetic.get()))

    def _run_calculate_phase_size(self):
        if self._ensure_loaded():
            return
        v = self._visualise
        it = self._phase_iter.get()
        self._run_viz(lambda: v.calculate_phase_size(iteration=it))

    def _run_plot_h(self):
        if self._ensure_loaded():
            return
        self._run_viz(self._visualise.plot_h)

    def _run_plot_plane0_product_tracking(self):
        if self._ensure_loaded():
            return
        self._run_viz(self._visualise.plot_plane0_product_tracking)

    def run(self):
        self.root.mainloop()


def main():
    app = VisualisationApp()
    app.run()


if __name__ == "__main__":
    main()
