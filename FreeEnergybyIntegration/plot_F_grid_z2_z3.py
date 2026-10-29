#!/usr/bin/env python3
"""Both transitions' one-dimensional free-energy profiles in one 2x2 figure.

    top row    : F_2(z_2)            at Wi = 0.02 and 1.32   (1-trans-z2)
    bottom row : F_23(z_2=0, z_3)    at Wi = 1.32 and 5.28   (2-trans-z3)

Wi = beta * 4321 / 45000 (twice the earlier labels; beta unchanged).

Panel arrangement follows
``examples/UIDD_polymer_dynamics_wi/2-trans-z3/F_z2_z3_grid_seed1.png``: the
same four Wi values, titles carrying only ``Wi``, free-energy y labels.  Its
canvas size, subplot spacing, font ladder and DPI are NOT copied here -- they
are parsed out of that figure's own scripts at run time by
``reference_style()`` and printed on every run, because they have already
changed more than once.  Do not restate them in this docstring: read the run's
own output, or README_F_grid.md, for the values currently in force.

The quantity is the same in all four panels: each curve is shifted to its own
minimum,

    F(z) = max_z Q(z) - Q(z),   Q_n = log(Z_n)/n,  Q_inf = continuum objective,

so every panel starts from 0 at its own minimiser.  Note this is a *per bead*
relative free energy in units of log Z per step; the learned-potential figure it
is styled after plots a different normalisation in k_B T, so the two figures are
comparable in shape, not in absolute height.

The bottom row is loaded through ``2-trans-z3/profile_targets`` own module, so
the ESS masking, the published maximizer-set edges and the plateau guard cannot
drift away from the standalone z3 figure.  The top row is loaded here; it is
cross-checked against the standalone z2 figure's CSV at Wi = 1.32 when that CSV exists.

    python plot_F_grid_z2_z3.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
Z2_DIR = HERE / "1-trans-z2" / "wi_001_10"
Z3_PLOT = HERE / "2-trans-z3" / "profile_targets" / "plot_profiles_wi_0p66_2p64.py"
Z2_REFERENCE_CSV = HERE / "1-trans-z2" / "F_z_wi_0p10_0p66.csv"

# The figure this one is styled to match.  Its geometry, type scale and raster
# size are READ FROM IT at run time rather than copied here: it is under active
# edit (figsize 15x10 -> 15x10.5, label 16 -> 24, title 17 -> 25.5 and
# 1868x1306 -> 1883x1393 all changed after this script was first written), and
# a copied constant silently goes stale while a parsed one cannot.
REF_DIR = ROOT / "examples" / "UIDD_polymer_dynamics_wi" / "2-trans-z3"
REF_GRID = REF_DIR / "plot_surface_grid.py"
REF_PANEL = REF_DIR / "plot_z3_surface.py"
REF_PNG = REF_DIR / "F_z2_z3_grid_seed1.png"

Z2_WI = (0.02, 1.32)
Z3_WI = (1.32, 5.28)

# The legend is the one element the reference has no opinion on -- it draws a
# single curve per panel and needs none.  Sized between its tick and label.
LEGEND_FONTSIZE = 18

# One colour per transition, taken from the reference figure's own CURVE dicts:
# matplotlib default C0 (#1f77b4) for the z2 row and black for the z3 row;
# kept in step with 1-trans-z2/plot_z2_z3_surface.py and
# 2-trans-z3/plot_z3_surface.py.  Within a row the four system
# sizes are separated only by the standing alpha ladder and marker shapes.
Z2_COLOR = "#1f77b4"            # matplotlib default C0
Z3_COLOR = "k"
SERIES = (
    ("n8", r"$n=8$", 8, 0.20, "o"),
    ("n16", r"$n=16$", 16, 0.35, "^"),
    ("n32", r"$n=32$", 32, 0.50, "s"),
    ("ninf", r"$n\to\infty$", None, 1.00, "D"),
)


class DataNotReady(RuntimeError):
    """A required input has not been atomically published yet."""


def _literal(node):
    """Evaluate a literal or a flat ``dict(a=1, b=2)`` call from the source."""
    import ast
    if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "dict":
        return {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}
    return ast.literal_eval(node)


def read_reference_constants(path: Path, names) -> dict:
    """Pull top-level assignments out of a sibling script without importing it.

    Importing the reference would execute its data loading and plotting
    machinery; all that is wanted here is a handful of style constants.
    """
    import ast
    if not path.exists():
        raise DataNotReady(f"missing the reference script {path}")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: dict = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in names:
                found[target.id] = _literal(node.value)
    missing = sorted(set(names) - set(found))
    if missing:
        raise DataNotReady(f"{path.name} no longer defines {missing}")
    return found


def reference_style() -> dict:
    """Geometry, type scale and pinned raster size, taken from the reference."""
    grid = read_reference_constants(
        REF_GRID, {"FIGSIZE", "SPACING", "PNG_DPI", "VECTOR_DPI"})
    panel = read_reference_constants(REF_PANEL, {"FS"})
    fs = dict(panel["FS"])
    for key in ("tick", "label", "title"):
        if key not in fs:
            raise DataNotReady(f"{REF_PANEL.name}: FS has no '{key}'")
    fs["legend"] = LEGEND_FONTSIZE
    if not REF_PNG.exists():
        raise DataNotReady(f"missing the reference raster {REF_PNG}")
    from PIL import Image
    with Image.open(REF_PNG) as image:
        target_pixels = tuple(image.size)
    return dict(figsize=tuple(grid["FIGSIZE"]), spacing=dict(grid["SPACING"]),
                png_dpi=int(grid["PNG_DPI"]), vector_dpi=int(grid["VECTOR_DPI"]),
                fs=fs, target_pixels=target_pixels)


def load_z3_module():
    """Import the z3 profile script by path (its folder name has a hyphen)."""
    spec = importlib.util.spec_from_file_location(Z3_PLOT.stem, Z3_PLOT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if tuple(module.TARGET_WI) != Z3_WI:
        raise ValueError(f"{Z3_PLOT.name} targets {module.TARGET_WI}, not {Z3_WI}")
    return module


def z2_mc_path(n: int, wi: float) -> Path:
    """Fixed-point MC result of ``1-trans-z2/fixed_points/run_mc_fixed.py``."""
    return HERE / "1-trans-z2" / "fixed_points" / "mc" / f"n{n}_wi_{wi:.6f}.npz"


def load_z2_series() -> list[dict]:
    """The four z2 curves at both Wi, each shifted to its own minimum."""
    z: np.ndarray | None = None
    out: list[dict] = []
    for key, label, n, alpha, marker in SERIES:
        profiles: dict[float, np.ndarray] = {}
        if key == "ninf":
            path = Z2_DIR / "data" / "ninf" / "ninf_wi_001_10.npz"
            if not path.exists():
                raise DataNotReady(f"missing {path.name}")
            with np.load(path, allow_pickle=False) as data:
                wi_values = np.asarray(data["wi_values"], dtype=float)
                here = np.asarray(data["z_grid"], dtype=float)
                completed = np.asarray(data["completed"], dtype=bool)
                f_all = np.asarray(data["F_all"], dtype=float)
                for target in Z2_WI:
                    index = int(np.argmin(np.abs(wi_values - target)))
                    if not np.isclose(wi_values[index], target, rtol=0.0, atol=5e-13):
                        raise ValueError(f"{path.name}: Wi={target:g} is absent")
                    if not completed[index]:
                        raise DataNotReady(f"ninf: Wi={target:.2f} is not complete")
                    q = f_all[index]
                    if not np.isfinite(q).all():
                        raise DataNotReady(f"ninf: Wi={target:.2f} is not finite")
                    profiles[target] = q.max() - q
        else:
            for target in Z2_WI:
                path = z2_mc_path(n, target)
                if not path.exists():
                    raise DataNotReady(f"missing {path.name}")
                with np.load(path, allow_pickle=False) as data:
                    if not bool(np.asarray(data["complete"]).item()):
                        raise DataNotReady(f"{path.name} is incomplete")
                    if int(np.asarray(data["n"]).item()) != n:
                        raise ValueError(f"{path.name} is not an n={n} result")
                    wi = float(np.asarray(data["wi"]).item())
                    if not np.isclose(wi, target, rtol=0.0, atol=5e-13):
                        raise ValueError(f"{path.name} holds Wi={wi:.12g}")
                    here = np.asarray(data["z_grid"], dtype=float)
                    q = np.asarray(data["log_Z"], dtype=float) / n
                if not np.isfinite(q).all():
                    raise DataNotReady(f"{path.name} is not finite")
                profiles[target] = q.max() - q
        if z is None:
            z = here
        elif not np.array_equal(z, here):
            raise ValueError("The z2 inputs do not share a z grid")
        out.append(dict(key=key, label=label, z=z, alpha=alpha, marker=marker,
                        profiles=profiles))
    return out


def check_z2_against_published(series: list[dict]) -> None:
    """The right-hand z2 column must reproduce the standalone z2 figure's CSV, if present."""
    if not Z2_REFERENCE_CSV.exists():
        return
    table = np.genfromtxt(Z2_REFERENCE_CSV, delimiter=",", names=True)
    rows = np.isclose(table["Wi"], Z2_WI[1], rtol=0.0, atol=1e-12)
    published = {key: table[f"U_{key}"][rows] for key, *_ in SERIES}
    for item in series:
        drawn = item["profiles"][Z2_WI[1]]
        reference = published[item["key"]]
        if drawn.shape != reference.shape:
            raise ValueError(f"z2 {item['key']}: grid differs from the published CSV")
        gap = float(np.max(np.abs(drawn - reference)))
        if gap > 5e-9:
            raise ValueError(
                f"z2 {item['key']} at Wi={Z2_WI[1]:g} differs from "
                f"{Z2_REFERENCE_CSV.name} by {gap:.3g}"
            )


def finite_minima(potential: np.ndarray) -> np.ndarray:
    """Indices of the discrete minima of a finite-n profile."""
    finite = np.flatnonzero(np.isfinite(potential))
    floor = float(np.min(potential[finite]))
    tolerance = max(1e-10, 1e-8 * max(float(np.nanmax(potential)), 1.0))
    return finite[np.abs(potential[finite] - floor) <= tolerance]


def row_ymax(series: list[dict], target: float) -> float:
    """The largest drawn value in one panel, before any y clipping."""
    return max(float(np.nanmax(item["profiles"][target])) for item in series)


def draw_row(axes, series: list[dict], targets, z, xlabel: str, ylabel: str,
             edges: dict | None, z3mod, color: str, fs: dict) -> None:
    """One row of two panels; ``edges`` is set only for the degenerate z3 row.

    Both panels share the y range of the *right* panel, so the pair reads
    against one scale.  That deliberately clips the left panel's tails, which
    run to 1.54 (z2) and 1.60 (z3) against row maxima of 0.59 and 0.86; the
    wells being compared all sit near y=0, so nothing of interest is lost.
    """
    ymax = row_ymax(series, targets[1])
    for ax, target in zip(axes, targets):
        low = high = 0.0
        if edges is not None:
            low, high = edges[target]
            if high > low:
                ax.axvspan(low, high, color=color,
                           alpha=z3mod.SET_BAND_ALPHA, lw=0, zorder=0)
        for item in series:
            potential = item["profiles"][target]
            ax.plot(z, potential, "-", color=color, alpha=item["alpha"],
                    lw=2.6, label=item["label"], zorder=2)
            if edges is not None and item["key"] == "ninf":
                if high > low:
                    ax.plot([low, high], [0.0, 0.0], "-", color=color,
                            alpha=item["alpha"], lw=z3mod.SET_BAR_LW,
                            solid_capstyle="butt", zorder=3)
                ends = np.unique([low, high])
                ax.plot(ends, np.zeros_like(ends), item["marker"], ms=8.5,
                        color=color, alpha=item["alpha"], lw=0, zorder=4)
                continue
            idx = finite_minima(potential)
            ax.plot(z[idx], potential[idx], item["marker"], ms=8.5,
                    color=color, alpha=item["alpha"], zorder=4)

        ax.axvline(0.0, ls=":", color="k", lw=0.8, zorder=2)
        ax.set_xlim(float(z[0]), float(z[-1]))
        ax.set_ylim(-0.035 * ymax, 1.03 * ymax if ymax else 1.0)
        ax.set_xlabel(xlabel, fontsize=fs["label"])
        ax.set_ylabel(ylabel, fontsize=fs["label"])
        ax.set_title(rf"$\mathrm{{Wi}}={target:g}$", fontsize=fs["title"])
        ax.tick_params(labelsize=fs["tick"])
        ax.grid(alpha=0.3)
        # Every panel is a U-shaped well, so both upper corners carry a steep
        # tail; upper centre is the only reliably empty region at this
        # legend size, and it is empty in all four panels.
        ax.legend(fontsize=fs["legend"], loc="upper center",
                  framealpha=0.92)


def atomic_savefig(fig: plt.Figure, stem: Path, style: dict) -> list[Path]:
    """Write png/pdf/svg of one figure, all framed by the same bounding box.

    The raster is pinned to the reference figure's exact pixel size and the
    vectors inherit the same box, so all three share identical framing and
    physical size (target_pixels / png_dpi inches).
    """
    width, height = (value / style["png_dpi"] for value in style["target_pixels"])
    # These tick labels are wider than the reference's, so the tight crop can
    # exceed the pinned raster.  Shrink the canvas until it fits rather than
    # clipping a y label off the edge; type sizes are in points and so are not
    # touched by this.
    for _ in range(8):
        fig.canvas.draw()
        tight = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.1)
        scale = min(width / tight.width, height / tight.height)
        if scale >= 1.0:
            break
        fig.set_size_inches(fig.get_size_inches() * scale * 0.999, forward=True)
    else:
        raise ValueError(
            f"content still needs {tight.width * style['png_dpi']:.0f} x "
            f"{tight.height * style['png_dpi']:.0f} px after shrinking"
        )
    bbox = Bbox.from_bounds(
        tight.x0 - (width - tight.width) / 2,
        tight.y0 - (height - tight.height) / 2,
        width,
        height,
    )
    written: list[Path] = []
    for suffix, dpi in (("png", style["png_dpi"]),
                        ("pdf", style["vector_dpi"]),
                        ("svg", style["vector_dpi"])):
        path = stem.with_suffix(f".{suffix}")
        temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.{suffix}")
        try:
            fig.savefig(temporary, dpi=dpi, bbox_inches=bbox, pad_inches=0)
            os.replace(temporary, path)
        finally:
            if temporary.exists():
                temporary.unlink()
        written.append(path)
    return written


def atomic_savetxt(path: Path, array: np.ndarray, header: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            np.savetxt(handle, array, delimiter=",", header=header,
                       comments="", fmt="%.10g")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> tuple[list[Path], Path]:
    z3mod = load_z3_module()
    z2_series = load_z2_series()
    check_z2_against_published(z2_series)
    if Z2_REFERENCE_CSV.exists():
        print(f"  z2 Wi={Z2_WI[1]:g} column reproduces {Z2_REFERENCE_CSV.name}")
    z3_series = [z3mod.load_series(*spec) for spec in z3mod.SERIES]

    z2 = z2_series[0]["z"]
    z3 = z3_series[0]["z3"]
    if any(not np.allclose(item["z3"], z3, rtol=0.0, atol=1e-12)
           for item in z3_series[1:]):
        raise ValueError("The z3 inputs do not share a grid")
    edges = next(item["edges"] for item in z3_series if item["key"] == "ninf")

    style = reference_style()
    fs = style["fs"]
    print(f"  style read from {REF_GRID.name} / {REF_PANEL.name}: "
          f"figsize={style['figsize']}, tick={fs['tick']}, label={fs['label']}, "
          f"title={fs['title']}, raster={style['target_pixels'][0]}x"
          f"{style['target_pixels'][1]}")
    # Journals routinely refuse Matplotlib's default Type 3 fonts; the
    # reference embeds TrueType instead, so this one does too.
    matplotlib.rcParams["pdf.fonttype"] = 42

    fig, axes = plt.subplots(2, 2, figsize=style["figsize"])
    fig.subplots_adjust(**style["spacing"])
    draw_row(axes[0], z2_series, Z2_WI, z2, r"$z_2$", r"$F_{2,n}(z_2)$",
             None, z3mod, Z2_COLOR, fs)
    draw_row(axes[1], z3_series, Z3_WI, z3, r"$z_3$",
             r"$F_{23,n}(z_2{=}0,z_3)$", edges, z3mod, Z3_COLOR, fs)

    written = atomic_savefig(fig, HERE / "F_z2_z3_grid", style)
    plt.close(fig)

    rows = []
    for panel, series, targets, grid in (("z2", z2_series, Z2_WI, z2),
                                         ("z3", z3_series, Z3_WI, z3)):
        for target in targets:
            rows.append(np.column_stack([
                np.full(len(grid), 2.0 if panel == "z2" else 3.0),
                np.full(len(grid), target), grid,
                *(item["profiles"][target] for item in series),
            ]))
    csv = HERE / "F_z2_z3_grid.csv"
    atomic_savetxt(csv, np.concatenate(rows, axis=0),
                   "panel,Wi,z,F_n8,F_n16,F_n32,F_ninf")
    for path in [*written, csv]:
        print(f"  -> {path}")
    return written, csv


if __name__ == "__main__":
    main()
