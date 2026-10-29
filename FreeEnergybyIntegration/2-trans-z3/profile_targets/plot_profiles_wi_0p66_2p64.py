#!/usr/bin/env python3
"""Relative-potential profiles in ``z3`` at ``Wi=1.32`` and ``Wi=5.28``.

This is the ``z3`` counterpart of ``1-trans-z2/F_z_wi_0p10_0p66.png``: two
panels straddling a transition, four system sizes per panel, each curve shifted
to its own minimum so the shapes are comparable,

    U(z3) = max_z3 Q(z3) - Q(z3),

with ``Q_n = log(Z_n)/n`` for MC and the continuum objective for ``n -> inf``.
With the flow convention ``Wi = beta * 4321 / 45000`` the two Wi values
straddle the ``z3`` plateau onset ``beta = 4*pi^2`` (``Wi = 3.7908``):
``Wi=1.32`` has a single maximizer at ``z3=0``, while ``Wi=5.28`` is above the
onset, where the continuum maximizer is a whole interval rather than a point.

Inputs are the fixed-point MC results of ``../fixed_points/run_mc_tilted_fixed.py``
(``../fixed_points/mc/n<n>_wi_<Wi>.npz``), the fine-grid continuum profiles from
``run_ninf_profile_wi_0p66_2p64.py`` and the maximizer-set edges of the
n -> infinity sweep.

Finite ``n`` is symmetrised in ``z3`` exactly as the sweep's maximizer plot
does, ``log Z <- log[(Z(z3) + Z(-z3))/2]``, and a grid cell whose paired ESS is
below 300 is left blank instead of being drawn or interpolated.
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
import numpy as np


HERE = Path(__file__).resolve().parent
SWEEP_DATA = HERE.parent / "wi_001_10" / "data"
FIXED_MC = HERE.parent / "fixed_points" / "mc"
NINF_DIR = HERE / "ninf"
PLOT_DIR = HERE.parent
TARGET_WI = (1.32, 5.28)
WI_TO_BETA = 45000.0 / 4321.0
BETA_PLATEAU_ONSET = 4.0 * np.pi**2
ESS_FLOOR = 300.0
# The continuum maximizer set is an extended object, so it needs a fill as well
# as a mark.  The parent maximizer figures shade their set at C0/0.12, but there
# the fill lies on white; here it lies *behind* four C0 curves whose faintest is
# alpha 0.20, and 0.12 swallows the n=8 structure.  0.08 keeps the extent
# obvious while leaving the thick bar to carry the claim.
SET_BAND_ALPHA = 0.08
SET_BAR_LW = 4.0
BASE_COLOR = "C0"
OUTPUT_DPI = 150
TARGET_PIXELS = (1874, 908)

SERIES = (
    ("n8", r"$n=8$", 8, 0.20, "o", "mc"),
    ("n16", r"$n=16$", 16, 0.35, "^", "mc"),
    ("n32", r"$n=32$", 32, 0.50, "s", "mc"),
    ("ninf", r"$n\to\infty$", None, 1.00, "D", "ninf"),
)


class DataNotReady(RuntimeError):
    """A required aggregate has not been atomically published yet."""


def atomic_savefig(fig: plt.Figure, path: Path,
                   target_size_px: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp.png")
    try:
        fig.canvas.draw()
        tight = fig.get_tightbbox(fig.canvas.get_renderer()).padded(0.1)
        width, height = (value / OUTPUT_DPI for value in target_size_px)
        bbox = Bbox.from_bounds(
            tight.x0 - (width - tight.width) / 2,
            tight.y0 - (height - tight.height) / 2,
            width,
            height,
        )
        fig.savefig(temporary, dpi=OUTPUT_DPI, bbox_inches=bbox, pad_inches=0)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


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


def load_mc_series(key: str, label: str, n: int, alpha: float,
                   marker: str) -> dict:
    rows = []
    for target in TARGET_WI:
        path = FIXED_MC / f"n{n}_wi_{target:.6f}.npz"
        if not path.exists():
            raise DataNotReady(f"missing {path.name}")
        with np.load(path, allow_pickle=False) as data:
            if int(np.asarray(data["n"]).item()) != n:
                raise ValueError(f"{path.name} is not an n={n} result")
            if not np.isclose(float(data["wi"]), target, rtol=0.0, atol=5e-13):
                raise ValueError(f"{path.name} holds Wi={float(data['wi']):g}")
            if not np.isclose(float(data["beta"]), target * WI_TO_BETA, rtol=0.0, atol=5e-10):
                raise ValueError(f"{path.name}: beta does not match Wi*45000/4321")
            rows.append((bool(data["complete"]), np.asarray(data["z3_grid"], dtype=float),
                         np.asarray(data["mc_logZ"], dtype=float),
                         np.asarray(data["paired_ess"], dtype=float)))
    if any(not np.array_equal(r[1], rows[0][1]) for r in rows):
        raise ValueError(f"n={n}: the fixed-point files do not share a z3 grid")
    wi_values = np.asarray(TARGET_WI, dtype=float)
    completed = np.array([r[0] for r in rows])
    z3 = rows[0][1]
    log_z = np.stack([r[2] for r in rows])
    paired_ess = np.stack([r[3] for r in rows])

    if not np.allclose(z3, -z3[::-1], rtol=0.0, atol=1e-12):
        raise ValueError(f"{path.name}: z3 grid is not mirror symmetric")

    profiles: dict[float, np.ndarray] = {}
    ess: dict[float, np.ndarray] = {}
    for target in TARGET_WI:
        index = int(np.argmin(np.abs(wi_values - target)))
        if not np.isclose(wi_values[index], target, rtol=0.0, atol=5e-13):
            raise ValueError(f"{path.name}: Wi={target:g} is absent")
        if not completed[index]:
            raise DataNotReady(f"{key}: Wi={target:.2f} is not complete")
        row = log_z[index]
        if not np.isfinite(row).all():
            raise DataNotReady(f"{key}: Wi={target:.2f} has a non-finite profile")
        # The runner already symmetrises at the sample-weight level, so
        # mc_logZ is bitwise even in z3 and this is an idempotent restatement
        # of that contract, not a correction.  (mc_logZ_raw is the
        # un-symmetrised profile; its asymmetry reaches 0.61 for n=32.)
        symmetrised = np.logaddexp(row, row[::-1]) - np.log(2.0)
        q = symmetrised / n
        potential = q.max() - q
        # A cell is admissible only when both mirror halves are well sampled.
        reliable = np.minimum(paired_ess[index], paired_ess[index][::-1]) >= ESS_FLOOR
        profiles[target] = np.where(reliable, potential, np.nan)
        ess[target] = paired_ess[index]
    return dict(key=key, label=label, z3=z3, alpha=alpha, marker=marker,
                profiles=profiles, paired_ess=ess, source=path)


def sweep_edges() -> dict[float, tuple[float, float]]:
    """The published continuum maximizer set, exactly 0 below the onset.

    Above the onset the continuum maximizer is a whole interval, because
    cyclically time-translating the optimal slope profile leaves both endpoint
    constraints and the objective invariant while sliding z3 across
    ``[2 min w*, 2 max w*]``.  The sweep stores that extent as
    ``edge_lo/edge_hi = +-2||w*||_inf``; this figure draws it rather than a
    level set of its own profile, so that it cannot contradict the parent
    maximizer figure.

    A level set is not usable here: the objective leaves the plateau
    quadratically, so on the 0.01 z3 grid every tolerance between 1e-7 and
    1e-5 returns +-0.260 -- most of a grid cell short of the true edge --
    while anything tighter than the 8.6e-8 solver noise floor returns the
    spurious, disconnected pair +-0.18.
    """
    path = SWEEP_DATA / "ninf" / "ninf_joint_z3_wi_001_10.npz"
    if not path.exists():
        raise DataNotReady(f"missing {path.name}")
    with np.load(path, allow_pickle=False) as data:
        wi_values = np.asarray(data["wi_values"], dtype=float)
        edge_lo = np.asarray(data["edge_lo"], dtype=float)
        edge_hi = np.asarray(data["edge_hi"], dtype=float)
        above = np.asarray(data["above_onset"], dtype=bool)
        completed = np.asarray(data["completed"], dtype=bool)
    edges: dict[float, tuple[float, float]] = {}
    for target in TARGET_WI:
        index = int(np.argmin(np.abs(wi_values - target)))
        if not np.isclose(wi_values[index], target, rtol=0.0, atol=5e-13):
            raise ValueError(f"{path.name}: Wi={target:g} is absent")
        if not completed[index]:
            raise DataNotReady(f"ninf edges: Wi={target:.2f} is not complete")
        low, high = float(edge_lo[index]), float(edge_hi[index])
        if not (np.isfinite(low) and np.isfinite(high)) or abs(low + high) > 1e-12:
            raise ValueError(f"{path.name}: asymmetric maximizer set at "
                             f"Wi={target:g}")
        # The drawn geometry and the stored onset flag must never disagree.
        if (high > low) != bool(above[index]):
            raise ValueError(f"{path.name}: edges and above_onset disagree at "
                             f"Wi={target:.2f}")
        edges[target] = (low, high)
    return edges


def load_ninf_series(key: str, label: str, alpha: float, marker: str) -> dict:
    profiles: dict[float, np.ndarray] = {}
    edges: dict[float, tuple[float, float]] = {}
    edge_table = sweep_edges()
    z3: np.ndarray | None = None
    for target in TARGET_WI:
        tag = f"{target:.2f}".replace(".", "p")
        path = NINF_DIR / f"ninf_profile_wi_{tag}.npz"
        if not path.exists():
            raise DataNotReady(
                f"missing {path.name}; run run_ninf_profile_wi_0p66_2p64.py"
            )
        with np.load(path, allow_pickle=False) as data:
            if not bool(np.asarray(data["complete"]).item()):
                raise DataNotReady(f"{path.name} is incomplete")
            wi = float(np.asarray(data["wi"]).item())
            if not np.isclose(wi, target, rtol=0.0, atol=5e-13):
                raise ValueError(f"{path.name} holds Wi={wi:.12g}, not {target:g}")
            here = np.asarray(data["z3_grid"], dtype=float)
            potential = np.asarray(data["potential"], dtype=float)
            above_here = bool(np.asarray(data["above_onset"]).item())
        if not np.isfinite(potential).all():
            raise DataNotReady(f"{path.name} has a non-finite profile")
        if z3 is None:
            z3 = here
        elif not np.allclose(z3, here, rtol=0.0, atol=1e-12):
            raise ValueError("The continuum profiles do not share a z3 grid")
        profiles[target] = potential
        low, high = edge_table[target]
        if (high > low) != above_here:
            raise ValueError(f"{path.name}: above_onset disagrees with the "
                             f"sweep's maximizer set")
        # The edges come from a different file, so check they still land on
        # this profile's own plateau rather than on its rising flank.
        span = float(np.max(potential))
        if float(np.interp(high, here, potential)) > 1e-3 * span:
            raise ValueError(f"{path.name}: the published edge {high:+.6f} is "
                             f"not on this profile's plateau")
        edges[target] = (low, high)
    assert z3 is not None
    return dict(key=key, label=label, z3=z3, alpha=alpha, marker=marker,
                profiles=profiles, edges=edges, source=NINF_DIR)


def load_series(key: str, label: str, n: int | None, alpha: float,
                marker: str, kind: str) -> dict:
    if kind == "mc":
        assert n is not None
        return load_mc_series(key, label, n, alpha, marker)
    return load_ninf_series(key, label, alpha, marker)


def marker_positions(item: dict, target: float) -> np.ndarray:
    """Indices of the discrete minima of a *finite-n* profile.

    Never call this for the continuum series: its bottom is flat to the 1e-7
    solver noise floor, so a grid argmin there returns whichever cell the
    residual happened to favour, not a minimizer.
    """
    potential = item["profiles"][target]
    finite = np.flatnonzero(np.isfinite(potential))
    floor = float(np.min(potential[finite]))
    tolerance = max(1e-10, 1e-8 * max(float(np.nanmax(potential)), 1.0))
    return finite[np.abs(potential[finite] - floor) <= tolerance]


def main() -> tuple[Path, Path]:
    series = [load_series(*spec) for spec in SERIES]
    z3 = series[0]["z3"]
    # The mirrored continuum half-grid differs from the MC grid only by the
    # last bit of the negative cells, so compare numerically, not bitwise.
    if any(not np.allclose(item["z3"], z3, rtol=0.0, atol=1e-12)
           for item in series[1:]):
        raise ValueError("The four inputs do not share a z3 grid")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    csv_rows: list[np.ndarray] = []
    for ax, target in zip(axes, TARGET_WI):
        beta = target * WI_TO_BETA
        stage = ("before" if beta < BETA_PLATEAU_ONSET else "after") + " transition"
        ymax = 0.0
        low, high = next(item["edges"][target] for item in series
                         if item["key"] == "ninf")
        if high > low:
            # Above the onset the continuum maximizer is a whole interval.
            ax.axvspan(low, high, color=BASE_COLOR, alpha=SET_BAND_ALPHA,
                       lw=0, zorder=0)
        for item in series:
            potential = item["profiles"][target]
            ymax = max(ymax, float(np.nanmax(potential)))
            style = dict(color=BASE_COLOR, alpha=item["alpha"], lw=2.6)
            ax.plot(z3, potential, "-", label=item["label"], zorder=2, **style)
            if item["key"] == "ninf":
                # The maximizer set carries no legend row on purpose.  A legend
                # entry would have to name a *property of one curve* alongside
                # four series names, and its swatch (a short C0 bar, or a lone
                # diamond) is not separable from the n->infinity series' own
                # line and marker.  The band, the bar and the endpoint diamonds
                # are explained in the figure caption instead -- see the
                # "图注 / Caption" section of README.md, which is kept in sync
                # with the numbers this script prints.
                if high > low:
                    # One connected segment, not two markers: every z3 in here
                    # is a maximizer, so the mark has to be extended too.
                    ax.plot([low, high], [0.0, 0.0], "-", color=BASE_COLOR,
                            alpha=item["alpha"], lw=SET_BAR_LW,
                            solid_capstyle="butt", zorder=3)
                # np.unique collapses (-0.0, 0.0) to one point below the onset.
                ends = np.unique([low, high])
                ax.plot(ends, np.zeros_like(ends), item["marker"], ms=8.5,
                        color=BASE_COLOR, alpha=item["alpha"], lw=0, zorder=4)
                continue
            idx = marker_positions(item, target)
            ax.plot(z3[idx], potential[idx], item["marker"], ms=8.5,
                    color=BASE_COLOR, alpha=item["alpha"], zorder=4)

        ax.axvline(0.0, ls=":", color="k", lw=0.8, zorder=2)
        ax.set_xlim(float(z3[0]), float(z3[-1]))
        ax.set_ylim(-0.035 * ymax, 1.03 * ymax if ymax else 1.0)
        ax.set_xlabel(r"$z_3$", fontsize=13)
        ax.set_ylabel(r"relative potential $U(z_3)-\min U$", fontsize=12)
        ax.set_title(rf"$Wi={target:.2f}$  ($\beta={beta:.6f}$; {stage})",
                     fontsize=14)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=10, loc="upper left", framealpha=0.92)

        csv_rows.append(np.column_stack([
            np.full(len(z3), target), np.full(len(z3), beta), z3,
            *(item["profiles"][target] for item in series),
        ]))

    fig.suptitle(
        r"One-dimensional relative-potential profiles in $z_3$: "
        r"$n=8,16,32,\;n\to\infty$",
        fontsize=14,
    )
    figure = PLOT_DIR / "F_z3_wi_1p32_5p28.png"
    atomic_savefig(fig, figure, TARGET_PIXELS)
    plt.close(fig)

    csv = PLOT_DIR / "F_z3_wi_1p32_5p28.csv"
    atomic_savetxt(csv, np.concatenate(csv_rows, axis=0),
                   "Wi,beta,z3,U_n8,U_n16,U_n32,U_ninf")
    for item in series:
        if item["key"] == "ninf":
            for target, (low, high) in item["edges"].items():
                kind = "interval" if high > low else "single point"
                print(f"  n->inf maximizer set at Wi={target:.2f}: "
                      f"[{low:+.7f}, {high:+.7f}]  ({kind}; sweep edge_lo/edge_hi)")
    print(f"  -> {figure}")
    print(f"  -> {csv}")
    return figure, csv


if __name__ == "__main__":
    main()
