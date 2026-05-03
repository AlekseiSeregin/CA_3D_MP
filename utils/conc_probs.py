import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox, RadioButtons, Button, Slider


EXACT_FORMULA_GE1_TEXT = (
    "P(X >= 1) = 1 - (1 - 9/n^2)^k"
)
EXACT_FORMULA_GE1_LATEX = (
    r"$P(X\geq1)=1-\left(1-\frac{9}{n^2}\right)^k$"
)
EXACT_FORMULA_TEXT = (
    "P(X >= 2) = 1 - (1 - 9/n^2)^k - k*(9/n^2)*(1 - 9/n^2)^(k - 1)"
)
EXACT_FORMULA_LATEX = (
    r"$P(X\geq2)=1-\left(1-\frac{9}{n^2}\right)^k-"
    r"k\frac{9}{n^2}\left(1-\frac{9}{n^2}\right)^{k-1}$"
)
EXACT_FORMULA_GE3_TEXT = (
    "P(X >= 3) = 1 - [ (1 - 9/n^2)^k + k*(9/n^2)*(1 - 9/n^2)^(k - 1) "
    "+ (k*(k-1)/2)*(9/n^2)^2*(1 - 9/n^2)^(k - 2) ]"
)
EXACT_FORMULA_GE3_LATEX = (
    r"$P(X\geq3)=1-\left[\left(1-\frac{9}{n^2}\right)^k + "
    r"k\frac{9}{n^2}\left(1-\frac{9}{n^2}\right)^{k-1} + "
    r"\frac{k(k-1)}{2}\left(\frac{9}{n^2}\right)^2\left(1-\frac{9}{n^2}\right)^{k-2}\right]$"
)
EXACT_FORMULA_GE4_TEXT = (
    "P(X >= 4) = 1 - [ (1 - 9/n^2)^k + k*(9/n^2)*(1 - 9/n^2)^(k - 1) "
    "+ (k*(k-1)/2)*(9/n^2)^2*(1 - 9/n^2)^(k - 2) "
    "+ (k*(k-1)*(k-2)/6)*(9/n^2)^3*(1 - 9/n^2)^(k - 3) ]"
)
EXACT_FORMULA_GE4_LATEX = (
    r"$P(X\geq4)=1-\left[\left(1-\frac{9}{n^2}\right)^k + "
    r"k\frac{9}{n^2}\left(1-\frac{9}{n^2}\right)^{k-1} + "
    r"\frac{k(k-1)}{2}\left(\frac{9}{n^2}\right)^2\left(1-\frac{9}{n^2}\right)^{k-2} + "
    r"\frac{k(k-1)(k-2)}{6}\left(\frac{9}{n^2}\right)^3\left(1-\frac{9}{n^2}\right)^{k-3}\right]$"
)


def eval_equation(expr: str, x: np.ndarray) -> np.ndarray:
    """
    Evaluate an equation string using x and selected numpy functions.
    Example expression: '0.2 * np.exp(-x/3) + 0.1'
    """
    safe_env = {
        "np": np,
        "x": x,
        "sin": np.sin,
        "cos": np.cos,
        "tan": np.tan,
        "exp": np.exp,
        "log": np.log,
        "sqrt": np.sqrt,
        "abs": np.abs,
        "pi": np.pi,
        "e": np.e,
    }
    return eval(expr, {"__builtins__": {}}, safe_env)


def exact_formula_prob_at_least_two(k: np.ndarray, n: float) -> np.ndarray:
    p = 9.0 / (n ** 2)
    return 1.0 - (1.0 - p) ** k - k * p * (1.0 - p) ** (k - 1.0)


def exact_formula_prob_at_least_one(k: np.ndarray, n: float) -> np.ndarray:
    p = 9.0 / (n ** 2)
    return 1.0 - (1.0 - p) ** k


def exact_formula_prob_at_least_three(k: np.ndarray, n: float) -> np.ndarray:
    p = 9.0 / (n ** 2)
    return (
        1.0
        - (
            (1.0 - p) ** k
            + k * p * (1.0 - p) ** (k - 1.0)
            + (k * (k - 1.0) / 2.0) * (p ** 2) * (1.0 - p) ** (k - 2.0)
        )
    )


def exact_formula_prob_at_least_four(k: np.ndarray, n: float) -> np.ndarray:
    p = 9.0 / (n ** 2)
    return (
        1.0
        - (
            (1.0 - p) ** k
            + k * p * (1.0 - p) ** (k - 1.0)
            + (k * (k - 1.0) / 2.0) * (p ** 2) * (1.0 - p) ** (k - 2.0)
            + (k * (k - 1.0) * (k - 2.0) / 6.0) * (p ** 3) * (1.0 - p) ** (k - 3.0)
        )
    )


EXACT_FUNCTIONS = {
    "ge1": (exact_formula_prob_at_least_one, EXACT_FORMULA_GE1_TEXT, EXACT_FORMULA_GE1_LATEX),
    "ge2": (exact_formula_prob_at_least_two, EXACT_FORMULA_TEXT, EXACT_FORMULA_LATEX),
    "ge3": (exact_formula_prob_at_least_three, EXACT_FORMULA_GE3_TEXT, EXACT_FORMULA_GE3_LATEX),
    "ge4": (exact_formula_prob_at_least_four, EXACT_FORMULA_GE4_TEXT, EXACT_FORMULA_GE4_LATEX),
}


def _compute_product_lines(
    f1_key: str,
    f2_key: str,
    n: float,
    c1min: float,
    c1max: float,
    c2min: float,
    c2max: float,
    k2_lines: int,
    points: int,
    result_scale: float = 1.0,
):
    func1 = EXACT_FUNCTIONS[f1_key][0]
    func2 = EXACT_FUNCTIONS[f2_key][0]
    x = np.linspace(c1min, c1max, points)
    k1 = x * (n ** 2)
    y1 = func1(k1, n)
    c2_values = np.linspace(c2min, c2max, k2_lines)
    line_series = []
    for c2_fixed in c2_values:
        k2_fixed = c2_fixed * (n ** 2)
        y2_scalar = float(func2(np.array([k2_fixed]), n)[0])
        line_series.append((c2_fixed, (y1 * y2_scalar) * result_scale))
    return x, line_series


def _plot_product_lines(ax, x, line_series, f2_key, n):
    ax.clear()
    for c2_fixed, y_line in line_series:
        ax.plot(x, y_line, linewidth=2, label=f"c2={c2_fixed:.3g}")
    ax.set_title(f"Product curves, n={n}")
    ax.set_xlabel("c1 = k1 / n^2")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend(title=f"{f2_key} fixed c2", fontsize=9)


def _compute_product_single_curve(
    f1_key: str,
    f2_key: str,
    n: float,
    c1min: float,
    c1max: float,
    c2: float,
    points: int,
    result_scale: float = 1.0,
):
    func1 = EXACT_FUNCTIONS[f1_key][0]
    func2 = EXACT_FUNCTIONS[f2_key][0]
    x = np.linspace(c1min, c1max, points)
    k1 = x * (n ** 2)
    y1 = func1(k1, n)
    k2 = c2 * (n ** 2)
    y2_scalar = float(func2(np.array([k2]), n)[0])
    y = (y1 * y2_scalar) * result_scale
    return x, y


def _plot_product_single_curve(ax, x, y, f1_key, f2_key, n, c2):
    ax.clear()
    ax.plot(x, y, linewidth=2, label=f"c2={c2:.4g}")
    ax.set_title(f"Product curve: {f1_key}(c1,n) * {f2_key}(c2,n), n={n}")
    ax.set_xlabel("c1 = k1 / n^2")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)


def _compute_total_vs_c2_curve(
    f1_key: str,
    f2_key: str,
    n: float,
    c1_const: float,
    c2min: float,
    c2max: float,
    points: int,
    result_scale: float = 1.0,
):
    func1 = EXACT_FUNCTIONS[f1_key][0]
    func2 = EXACT_FUNCTIONS[f2_key][0]
    c2_x = np.linspace(c2min, c2max, points)
    k1_const = c1_const * (n ** 2)
    y1_const = float(func1(np.array([k1_const]), n)[0])
    k2 = c2_x * (n ** 2)
    y2 = func2(k2, n)
    y_total = (y1_const * y2) * result_scale
    return c2_x, y_total


def _plot_total_vs_c2_curve(ax, c2_x, y_total, f1_key, f2_key, n, c1_const, c2_current):
    ax.clear()
    ax.plot(c2_x, y_total, linewidth=2, color="tab:orange", label=f"c1 fixed at {c1_const:.4g}")
    ax.axvline(c2_current, color="tab:red", linestyle="--", linewidth=1.5, label=f"current c2={c2_current:.4g}")
    ax.set_title(f"Total vs c2: {f1_key}(c1_const,n) * {f2_key}(c2,n), n={n}")
    ax.set_xlabel("c2 = k2 / n^2")
    ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)


def launch_interactive_product_ui(initial_args) -> None:
    fig = plt.figure(figsize=(12, 7))
    ax_plot = fig.add_axes([0.34, 0.56, 0.63, 0.36])
    ax_plot_c2 = fig.add_axes([0.34, 0.16, 0.63, 0.30])

    ax_f1 = fig.add_axes([0.03, 0.75, 0.12, 0.17])
    ax_f2 = fig.add_axes([0.17, 0.75, 0.12, 0.17])
    choices = ("ge1", "ge2", "ge3", "ge4")
    rb_f1 = RadioButtons(ax_f1, choices, active=choices.index(initial_args.f1))
    rb_f2 = RadioButtons(ax_f2, choices, active=choices.index(initial_args.f2))
    ax_f1.set_title("f1")
    ax_f2.set_title("f2")

    box_specs = [
        ("n", str(initial_args.n)),
        ("c1min", str(initial_args.c1min)),
        ("c1max", str(initial_args.c1max)),
        ("c2min", str(initial_args.c2min)),
        ("c2max", str(initial_args.c2max)),
        ("result_scale", str(initial_args.result_scale)),
        ("points", str(initial_args.points)),
    ]
    text_boxes = {}
    y0 = 0.62
    dy = 0.075
    for idx, (label, value) in enumerate(box_specs):
        ax_box = fig.add_axes([0.03, y0 - idx * dy, 0.26, 0.05])
        text_boxes[label] = TextBox(ax_box, f"{label}: ", initial=value)

    c2_lo = float(initial_args.c2min)
    c2_hi = float(initial_args.c2max)
    if c2_hi <= c2_lo:
        c2_hi = c2_lo + 1e-6
    c2_init = float(initial_args.c2)
    c2_init = min(max(c2_init, c2_lo), c2_hi)
    ax_slider = fig.add_axes([0.36, 0.07, 0.58, 0.04])
    c2_slider = Slider(ax_slider, "c2 = k2/n^2", c2_lo, c2_hi, valinit=c2_init)

    ax_apply = fig.add_axes([0.03, 0.06, 0.12, 0.05])
    btn_apply = Button(ax_apply, "Recalculate")
    status_txt = fig.text(0.03, 0.015, "", fontsize=9)

    state = {"f1": initial_args.f1, "f2": initial_args.f2}
    suppress_slider_callback = {"value": False}
    last_params = {
        "n": float(initial_args.n),
        "c1min": float(initial_args.c1min),
        "c1max": float(initial_args.c1max),
        "c2min": float(initial_args.c2min),
        "c2max": float(initial_args.c2max),
        "result_scale": float(initial_args.result_scale),
        "points": int(initial_args.points),
    }

    def _read_positive_float(key):
        return float(text_boxes[key].text.strip())

    def _read_positive_int(key):
        return int(float(text_boxes[key].text.strip()))

    def _draw_from_current_slider():
        p = last_params
        c2 = float(c2_slider.val)
        x, y = _compute_product_single_curve(
            state["f1"], state["f2"], p["n"], p["c1min"], p["c1max"], c2, p["points"], p["result_scale"]
        )
        _plot_product_single_curve(ax_plot, x, y, state["f1"], state["f2"], p["n"], c2)
        c2_x, y_total = _compute_total_vs_c2_curve(
            state["f1"], state["f2"], p["n"], p["c1max"], p["c2min"], p["c2max"], p["points"], p["result_scale"]
        )
        _plot_total_vs_c2_curve(ax_plot_c2, c2_x, y_total, state["f1"], state["f2"], p["n"], p["c1max"], c2)
        status_txt.set_text(f"Updated. c2={c2:.5g}, scale={p['result_scale']:.5g}")
        status_txt.set_color("black")
        fig.canvas.draw_idle()

    def redraw(_event=None):
        try:
            state["f1"] = rb_f1.value_selected
            state["f2"] = rb_f2.value_selected
            n = _read_positive_float("n")
            c1min = _read_positive_float("c1min")
            c1max = _read_positive_float("c1max")
            c2min = _read_positive_float("c2min")
            c2max = _read_positive_float("c2max")
            result_scale = float(text_boxes["result_scale"].text.strip())
            points = _read_positive_int("points")
            if n <= 0:
                raise ValueError("n must be > 0")
            if points < 2:
                raise ValueError("points must be >= 2")
            if c1min < 0 or c1max < 0 or c2min < 0 or c2max < 0:
                raise ValueError("concentrations must be >= 0")
            if c2max <= c2min:
                raise ValueError("c2max must be greater than c2min")

            last_params.update(
                {
                    "n": float(n),
                    "c1min": float(c1min),
                    "c1max": float(c1max),
                    "c2min": float(c2min),
                    "c2max": float(c2max),
                    "result_scale": float(result_scale),
                    "points": int(points),
                }
            )

            # Update slider range dynamically from text boxes.
            c2_slider.valmin = c2min
            c2_slider.valmax = c2max
            c2_slider.ax.set_xlim(c2min, c2max)
            c2 = float(c2_slider.val)
            if c2 < c2min or c2 > c2max:
                c2_clamped = min(max(c2, c2min), c2max)
                suppress_slider_callback["value"] = True
                c2_slider.set_val(c2_clamped)
                suppress_slider_callback["value"] = False
            _draw_from_current_slider()
        except Exception as exc:
            status_txt.set_text(f"Input error: {exc}")
            status_txt.set_color("red")
            fig.canvas.draw_idle()

    def _on_slider_changed(_val):
        if suppress_slider_callback["value"]:
            return
        try:
            _draw_from_current_slider()
        except Exception as exc:
            status_txt.set_text(f"Input error: {exc}")
            status_txt.set_color("red")
            fig.canvas.draw_idle()

    rb_f1.on_clicked(redraw)
    rb_f2.on_clicked(redraw)
    c2_slider.on_changed(_on_slider_changed)
    for tb in text_boxes.values():
        tb.on_submit(redraw)
    btn_apply.on_clicked(redraw)

    redraw()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot a user-provided equation y(x) with Matplotlib."
    )
    parser.add_argument(
        "equation",
        nargs="?",
        help="Equation in x. Example: '0.2*np.exp(-x/3) + 0.1'",
    )
    parser.add_argument("--xmin", type=float, default=0.0, help="Minimum x value.")
    parser.add_argument("--xmax", type=float, default=10.0, help="Maximum x value.")
    parser.add_argument(
        "--points", type=int, default=500, help="Number of x points in the plot."
    )
    parser.add_argument(
        "--exact-formula-ge1",
        action="store_true",
        help="Plot the exact formula P(X>=1) versus k.",
    )
    parser.add_argument(
        "--exact-formula-ge2",
        action="store_true",
        help="Plot the exact formula P(X>=2) versus k.",
    )
    parser.add_argument(
        "--exact-formula-ge3",
        action="store_true",
        help="Plot the exact formula P(X>=3) (with 9/n^2) versus k.",
    )
    parser.add_argument(
        "--exact-formula-ge4",
        action="store_true",
        help="Plot the exact formula P(X>=4) (with 9/n^2) versus k.",
    )
    parser.add_argument(
        "--product-mode",
        action="store_true",
        help="Plot product of two exact formulas with separate k ranges.",
    )
    parser.add_argument(
        "--product-static",
        action="store_true",
        help="Use static product plot. Without this flag, product mode opens interactive UI.",
    )
    parser.add_argument(
        "--f1",
        choices=("ge1", "ge2", "ge3", "ge4"),
        default="ge2",
        help="First formula for product mode.",
    )
    parser.add_argument(
        "--f2",
        choices=("ge1", "ge2", "ge3", "ge4"),
        default="ge3",
        help="Second formula for product mode.",
    )
    parser.add_argument("--c1min", type=float, default=0.0, help="Minimum concentration c1=k1/n^2 for first formula.")
    parser.add_argument("--c1max", type=float, default=0.05, help="Maximum concentration c1=k1/n^2 for first formula.")
    parser.add_argument("--c2", type=float, default=None, help="Fixed concentration c2=k2/n^2 for interactive product slider initial value.")
    parser.add_argument("--c2min", type=float, default=0.0, help="Minimum concentration c2=k2/n^2 for second formula.")
    parser.add_argument("--c2max", type=float, default=0.05, help="Maximum concentration c2=k2/n^2 for second formula.")
    parser.add_argument("--result-scale", type=float, default=1.0, help="Global multiplier applied to the whole product result curve.")
    parser.add_argument(
        "--k2-lines",
        type=int,
        default=6,
        help="Number of fixed k2 values (lines) in product mode.",
    )
    parser.add_argument("--n", type=float, default=100.0, help="n value for exact formula.")
    parser.add_argument("--kmin", type=float, default=0.0, help="Minimum k for exact formula.")
    parser.add_argument("--kmax", type=float, default=500.0, help="Maximum k for exact formula.")
    parser.add_argument("--k1min", type=float, default=None, help="Legacy: min k1 for product mode (converted to c1).")
    parser.add_argument("--k1max", type=float, default=None, help="Legacy: max k1 for product mode (converted to c1).")
    parser.add_argument("--k2min", type=float, default=None, help="Legacy: min k2 for product mode (converted to c2).")
    parser.add_argument("--k2max", type=float, default=None, help="Legacy: max k2 for product mode (converted to c2).")
    args = parser.parse_args()

    # Backward compatibility for old k-based product arguments.
    if args.k1min is not None:
        args.c1min = float(args.k1min) / (args.n ** 2)
    if args.k1max is not None:
        args.c1max = float(args.k1max) / (args.n ** 2)
    if args.k2min is not None:
        args.c2min = float(args.k2min) / (args.n ** 2)
    if args.k2max is not None:
        args.c2max = float(args.k2max) / (args.n ** 2)
    if args.c2 is None:
        args.c2 = 0.5 * (args.c2min + args.c2max)

    if args.product_mode and not args.product_static:
        launch_interactive_product_ui(args)
        return

    if args.product_mode:
        if args.c1min < 0 or args.c1max < 0 or args.c2min < 0 or args.c2max < 0:
            raise ValueError("Product-mode concentrations must be >= 0.")
        _, text1, _ = EXACT_FUNCTIONS[args.f1]
        _, text2, _ = EXACT_FUNCTIONS[args.f2]
        n_lines = max(1, int(args.k2_lines))
        x, line_series = _compute_product_lines(
            args.f1, args.f2, args.n, args.c1min, args.c1max, args.c2min, args.c2max, n_lines, args.points, args.result_scale
        )
        print("Product mode:")
        print(f"f1 = {args.f1}: {text1}")
        print(f"f2 = {args.f2}: {text2}")
        print(f"Shared n = {args.n}")
        print(f"c1 range = [{args.c1min}, {args.c1max}]")
        print(f"c2 fixed values from [{args.c2min}, {args.c2max}], lines = {n_lines}")
        print(f"result_scale = {args.result_scale}")
        plot_title = f"Product: {args.f1}(k1,n) * {args.f2}(k2,n), n={args.n}"
        x_label = "c1 = k1 / n^2"
        formula_latex = None
    elif args.exact_formula_ge4:
        print("Exact formula:")
        print(EXACT_FORMULA_GE4_TEXT)
        x = np.linspace(args.kmin, args.kmax, args.points)
        y = exact_formula_prob_at_least_four(x, args.n)
        plot_title = f"Exact formula P(X >= 4), n={args.n}"
        x_label = "k"
        formula_latex = EXACT_FORMULA_GE4_LATEX
    elif args.exact_formula_ge1:
        print("Exact formula:")
        print(EXACT_FORMULA_GE1_TEXT)
        x = np.linspace(args.kmin, args.kmax, args.points)
        y = exact_formula_prob_at_least_one(x, args.n)
        plot_title = f"Exact formula P(X >= 1), n={args.n}"
        x_label = "k"
        formula_latex = EXACT_FORMULA_GE1_LATEX
    elif args.exact_formula_ge3:
        print("Exact formula:")
        print(EXACT_FORMULA_GE3_TEXT)
        x = np.linspace(args.kmin, args.kmax, args.points)
        y = exact_formula_prob_at_least_three(x, args.n)
        plot_title = f"Exact formula P(X >= 3), n={args.n}"
        x_label = "k"
        formula_latex = EXACT_FORMULA_GE3_LATEX
    elif args.exact_formula_ge2:
        print("Exact formula:")
        print(EXACT_FORMULA_TEXT)
        x = np.linspace(args.kmin, args.kmax, args.points)
        y = exact_formula_prob_at_least_two(x, args.n)
        plot_title = f"Exact formula, n={args.n}"
        x_label = "k"
        formula_latex = EXACT_FORMULA_LATEX
    else:
        equation = args.equation
        if not equation:
            equation = input("Enter equation in x: ").strip()
        x = np.linspace(args.xmin, args.xmax, args.points)
        y = eval_equation(equation, x)
        plot_title = f"y(x) = {equation}"
        x_label = "x"
        formula_latex = None

    plt.figure(figsize=(9, 5))
    if args.product_mode:
        _plot_product_lines(plt.gca(), x, line_series, args.f2, args.n)
    else:
        plt.plot(x, y, linewidth=2)
    plt.title(plot_title)
    plt.xlabel(x_label)
    plt.ylabel("y")
    if formula_latex is not None:
        plt.figtext(0.5, 0.01, formula_latex, ha="center", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout(rect=(0, 0.04, 1, 1))
    plt.show()


if __name__ == "__main__":
    main()