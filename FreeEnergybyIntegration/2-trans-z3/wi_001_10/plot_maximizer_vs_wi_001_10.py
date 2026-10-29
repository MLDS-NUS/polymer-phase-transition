#!/usr/bin/env python3
"""Plot the joint ``(z2=0, z3)`` maximizer branches on ``Wi=0.02..10``.

The input files are deliberately kept in this directory's ``data/`` tree so
the expanded sweep is self-contained.  Finite-n curves use a symmetrised
profile and only admit a candidate ``z3`` cell when *both* mirror cells have
paired ESS of at least 300.  The ``n -> infinity`` curve is a maximizer set,
so it is drawn as its lower and upper endpoints rather than as an arbitrary
single argmax.

Default inputs (written by the companion resumable runners) are::

    data/ninf/ninf_joint_z3_wi_001_10.npz
    data/mc_n{8,16,32}_joint_z3_wi_001_10.npz

Run after the aggregates are present::

    python3 plot_maximizer_vs_wi_001_10.py

``--allow-partial`` is intentionally opt-in: it is useful for inspecting a
running sweep, but the detached supervisor calls this script only after the
four final aggregates exist.
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HERE / "data"
WI_TO_BETA = 45000.0 / 4321.0
ESS_FLOOR = 300.0
THEORY_LEVEL_TOL = 1.0e-6
OUTPUT_DPI = 150
TARGET_PIXELS = (1630, 955)
# The small fractional pixel guards against Matplotlib truncating a binary
# floating-point representation one pixel below the requested raster width.
FIGSIZE = tuple((pixels + 0.01) / OUTPUT_DPI for pixels in TARGET_PIXELS)
BASE_COLOR = "C0"

MC_SPECS = (
    ("n8", r"$n=8$", 8, 0.20, "o"),
    ("n16", r"$n=16$", 16, 0.35, "^"),
    ("n32", r"$n=32$", 32, 0.50, "s"),
)
NINF_SPEC = ("ninf", r"$n\to\infty$ (maximizer-set edge)", 1.00, "D")


class DataLayoutError(ValueError):
    """Raised when an aggregate does not contain a compatible data layout."""


@dataclass
class MCSeries:
    key: str
    label: str
    wi: np.ndarray
    beta: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    complete: np.ndarray
    paired_ess_negative: np.ndarray
    paired_ess_positive: np.ndarray
    reliable: np.ndarray
    argmax_at_boundary: np.ndarray
    quality_pass: np.ndarray
    profile_stable: np.ndarray
    alpha: float
    marker: str
    source: Path
    source_keys: tuple[str, ...]


@dataclass
class TheorySeries:
    key: str
    label: str
    wi: np.ndarray
    beta: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    complete: np.ndarray
    success: np.ndarray
    alpha: float
    marker: str
    source: Path
    source_keys: tuple[str, ...]
    endpoint_source: str


def _pick(data: np.lib.npyio.NpzFile, names: Iterable[str], *, required: bool,
          source: Path) -> np.ndarray | None:
    """Return the first compatible-key value, without silently guessing."""
    for name in names:
        if name in data.files:
            return np.asarray(data[name])
    if required:
        tried = ", ".join(names)
        raise DataLayoutError(f"{source}: missing required key (tried {tried})")
    return None


def _vector(value: np.ndarray | None, n_rows: int, name: str, *,
            dtype: type = float, default: float | bool | None = None) -> np.ndarray:
    """Coerce a scalar or one-value-per-frame field to a vector."""
    if value is None:
        if default is None:
            raise DataLayoutError(f"Missing required vector {name}")
        return np.full(n_rows, default, dtype=dtype)
    array = np.asarray(value)
    if array.ndim == 0:
        return np.full(n_rows, array.item(), dtype=dtype)
    array = np.squeeze(array)
    if array.ndim != 1 or len(array) != n_rows:
        raise DataLayoutError(
            f"{name} must be scalar or shape ({n_rows},), got {np.asarray(value).shape}"
        )
    return np.asarray(array, dtype=dtype)


def _frame_grid(value: np.ndarray, n_rows: int, n_grid: int, name: str) -> np.ndarray:
    """Accept either conventional frame-by-grid order or its transpose."""
    array = np.asarray(value, dtype=float)
    if array.ndim == 1 and n_rows == 1 and len(array) == n_grid:
        return array[None, :]
    if array.ndim != 2:
        raise DataLayoutError(f"{name} must be two-dimensional, got {array.shape}")
    if array.shape == (n_rows, n_grid):
        return array
    if array.shape == (n_grid, n_rows):
        return array.T
    raise DataLayoutError(
        f"{name} must have shape ({n_rows}, {n_grid}) or ({n_grid}, {n_rows}), "
        f"got {array.shape}"
    )


def _coordinates(data: np.lib.npyio.NpzFile, source: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load, validate, and sort Wi/beta coordinates from key aliases."""
    wi_raw = _pick(data, ("wi_values", "Wi", "wi", "wi_grid"),
                   required=False, source=source)
    beta_raw = _pick(data, ("beta_values", "beta", "betas", "beta_grid"),
                     required=False, source=source)
    if wi_raw is None and beta_raw is None:
        raise DataLayoutError(f"{source}: need wi_values/Wi or beta_values/beta")
    if wi_raw is None:
        beta = np.asarray(beta_raw, dtype=float).reshape(-1)
        wi = beta / WI_TO_BETA
    elif beta_raw is None:
        wi = np.asarray(wi_raw, dtype=float).reshape(-1)
        beta = wi * WI_TO_BETA
    else:
        wi = np.asarray(wi_raw, dtype=float).reshape(-1)
        beta = np.asarray(beta_raw, dtype=float).reshape(-1)
        if len(wi) != len(beta):
            raise DataLayoutError(f"{source}: Wi and beta lengths differ")
        if not np.allclose(beta, wi * WI_TO_BETA, rtol=2e-10, atol=2e-10):
            raise DataLayoutError(f"{source}: beta is inconsistent with Wi*45000/4321")
    if len(wi) == 0 or not np.isfinite(wi).all() or np.any(wi <= 0):
        raise DataLayoutError(f"{source}: Wi must be a non-empty, finite positive vector")
    order = np.argsort(wi)
    wi, beta = wi[order], beta[order]
    if np.any(np.diff(wi) <= 0):
        raise DataLayoutError(f"{source}: Wi values must be unique")
    return wi, beta, order


def _ordered_vector(value: np.ndarray | None, n_rows: int, order: np.ndarray,
                    name: str, *, dtype: type = float,
                    default: float | bool | None = None) -> np.ndarray:
    return _vector(value, n_rows, name, dtype=dtype, default=default)[order]


def _ordered_grid(value: np.ndarray, n_rows: int, n_grid: int, order: np.ndarray,
                  name: str) -> np.ndarray:
    return _frame_grid(value, n_rows, n_grid, name)[order]


def _final_or_partial(data_dir: Path, final_name: str, *, allow_partial: bool) -> Path:
    final = data_dir / final_name
    if final.exists():
        return final
    if allow_partial:
        partial = final.with_name(final.stem + "_partial" + final.suffix)
        if partial.exists():
            return partial
    hint = " (or --allow-partial for its partial aggregate)" if not allow_partial else ""
    raise FileNotFoundError(f"Missing {final}{hint}")


def _scalar_int(data: np.lib.npyio.NpzFile, names: Sequence[str], source: Path,
                fallback: int) -> int:
    raw = _pick(data, names, required=False, source=source)
    if raw is None:
        return fallback
    array = np.asarray(raw)
    if array.size != 1:
        raise DataLayoutError(f"{source}: {'/'.join(names)} must be scalar")
    return int(array.item())


def load_mc(key: str, label: str, n_expected: int, alpha: float, marker: str,
            data_dir: Path, *, allow_partial: bool) -> MCSeries:
    """Load an MC aggregate, enforcing paired-ESS filtering cell by cell."""
    source = _final_or_partial(data_dir, f"mc_n{n_expected}_joint_z3_wi_001_10.npz",
                               allow_partial=allow_partial)
    with np.load(source, allow_pickle=False) as data:
        wi, beta, order = _coordinates(data, source)
        n_rows = len(wi)
        z_grid_raw = _pick(data, ("z3_grid", "z_grid", "z_values", "z3_values"),
                            required=True, source=source)
        z_grid = np.asarray(z_grid_raw, dtype=float).reshape(-1)
        if len(z_grid) < 3 or not np.all(np.diff(z_grid) > 0):
            raise DataLayoutError(f"{source}: z3 grid must be increasing and contain >=3 points")
        n_value = _scalar_int(data, ("n", "n_value", "chain_length"), source, n_expected)
        if n_value != n_expected:
            raise DataLayoutError(f"{source}: reports n={n_value}, expected n={n_expected}")
        log_z = _ordered_grid(
            _pick(data, ("mc_logZ", "mc_log_z", "log_Z", "logZ", "log_z"),
                  required=True, source=source),
            n_rows, len(z_grid), order, "MC log-Z",
        )
        complete = _ordered_vector(
            _pick(data, ("completed", "complete", "is_complete"), required=False, source=source),
            n_rows, order, "completed", dtype=bool, default=True,
        )
        boundary = _ordered_vector(
            _pick(data, ("argmax_at_boundary", "max_at_boundary"), required=False, source=source),
            n_rows, order, "argmax_at_boundary", dtype=bool, default=False,
        )
        quality_pass = _ordered_vector(
            _pick(data, ("quality_pass", "quality_ok"), required=False, source=source),
            n_rows, order, "quality_pass", dtype=bool, default=False,
        )
        profile_stable = _ordered_vector(
            _pick(data, ("profile_stable", "stable_profile"), required=False, source=source),
            n_rows, order, "profile_stable", dtype=bool, default=False,
        )
        # Prefer the producer's sample-level paired ESS.  Raw ``ess`` is a
        # useful compatibility fallback only; it must then be mirror-minned.
        paired_ess_raw = _pick(data, ("paired_ess", "pairedESS",
                                      "effective_sample_size_paired"),
                               required=False, source=source)
        ess_raw = _pick(data, ("ess", "ESS", "effective_sample_size",
                              "effective_sample_sizes"), required=False, source=source)
        paired_min_raw = _pick(data, ("paired_ess_min", "min_paired_ess"),
                                      required=False, source=source)
        keys = tuple(sorted(data.files))

    center = int(np.argmin(np.abs(z_grid)))
    if not np.isclose(z_grid[center], 0.0, atol=1e-12):
        raise DataLayoutError(f"{source}: z3 grid must include zero for mirror pairing")
    # The z2=0 slice is exactly even in z3.  This reduces MC noise while
    # preserving an argmax, provided both cells in each pair are reliable.
    symmetrised = np.logaddexp(log_z, log_z[:, ::-1]) - np.log(2.0)
    finite_profile = np.isfinite(log_z) & np.isfinite(log_z[:, ::-1])

    if paired_ess_raw is not None:
        paired_array = np.asarray(paired_ess_raw)
        if paired_array.ndim == 1 or paired_array.ndim == 0:
            row_ess = _ordered_vector(paired_array, n_rows, order, "paired ESS", dtype=float)
            paired_ess = np.broadcast_to(row_ess[:, None], log_z.shape).copy()
        else:
            paired_ess = _ordered_grid(paired_array, n_rows, len(z_grid), order, "paired ESS")
    elif ess_raw is not None:
        ess_array = np.asarray(ess_raw)
        if ess_array.ndim == 1 or ess_array.ndim == 0:
            row_ess = _ordered_vector(ess_array, n_rows, order, "paired ESS", dtype=float)
            paired_ess = np.broadcast_to(row_ess[:, None], log_z.shape).copy()
        else:
            ess = _ordered_grid(ess_array, n_rows, len(z_grid), order, "ESS")
            paired_ess = np.minimum(ess, ess[:, ::-1])
    elif paired_min_raw is not None:
        row_ess = _ordered_vector(paired_min_raw, n_rows, order, "paired_ess_min", dtype=float)
        paired_ess = np.broadcast_to(row_ess[:, None], log_z.shape).copy()
    else:
        raise DataLayoutError(
            f"{source}: paired-ESS>=300 masking requires ess or paired_ess_min"
        )

    reliable_cell = finite_profile & np.isfinite(paired_ess) & (paired_ess >= ESS_FLOOR)
    reliable_cell &= complete[:, None]
    positive = z_grid >= -1e-14
    negative = z_grid <= 1e-14
    lower = np.full(n_rows, np.nan)
    upper = np.full(n_rows, np.nan)
    ess_negative = np.full(n_rows, np.nan)
    ess_positive = np.full(n_rows, np.nan)
    reliable = np.zeros(n_rows, dtype=bool)
    for row in range(n_rows):
        pos_index = np.flatnonzero(positive & reliable_cell[row])
        neg_index = np.flatnonzero(negative & reliable_cell[row])
        if len(pos_index):
            j = int(pos_index[np.argmax(symmetrised[row, pos_index])])
            upper[row] = z_grid[j]
            ess_positive[row] = paired_ess[row, j]
        if len(neg_index):
            j = int(neg_index[np.argmax(symmetrised[row, neg_index])])
            lower[row] = z_grid[j]
            ess_negative[row] = paired_ess[row, j]
        # A cell can clear the paired-ESS threshold while the producer still
        # considers the whole profile unsuitable (for example, because the
        # maximum is clipped by the z3 domain or independent half-samples do
        # not agree).  Do not turn such a diagnostic failure into a plotted
        # branch merely because another, lower free-energy cell is usable.
        reliable[row] = (
            np.isfinite(lower[row])
            and np.isfinite(upper[row])
            and bool(quality_pass[row])
            and bool(profile_stable[row])
            and not bool(boundary[row])
        )

    lower[~reliable] = np.nan
    upper[~reliable] = np.nan

    # Suppress numerical signed-zero artifacts in exported data and plot lines.
    lower[np.abs(lower) < 1e-14] = 0.0
    upper[np.abs(upper) < 1e-14] = 0.0
    return MCSeries(
        key=key, label=label, wi=wi, beta=beta, lower=lower, upper=upper,
        complete=complete, paired_ess_negative=ess_negative,
        paired_ess_positive=ess_positive, reliable=reliable,
        argmax_at_boundary=boundary, quality_pass=quality_pass,
        profile_stable=profile_stable, alpha=alpha, marker=marker,
        source=source, source_keys=keys,
    )


def _contiguous_run(mask: np.ndarray, containing: int) -> tuple[int, int]:
    """Find the contiguous true segment that includes ``containing``."""
    lo = containing
    hi = containing
    while lo > 0 and mask[lo - 1]:
        lo -= 1
    while hi + 1 < len(mask) and mask[hi + 1]:
        hi += 1
    return lo, hi


def _endpoints_from_profiles(data: np.lib.npyio.NpzFile, n_rows: int,
                             order: np.ndarray, source: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """Fallback for older full-profile theory outputs that lack edge arrays."""
    profile = _pick(data, ("F_all", "free_energy", "F", "phi"),
                    required=True, source=source)
    z_raw = _pick(data, ("z3_grid", "z_grid", "z_values"), required=True, source=source)
    z = np.asarray(z_raw, dtype=float).reshape(-1)
    rows = _ordered_grid(profile, n_rows, len(z), order, "theory profile")
    lower = np.full(n_rows, np.nan)
    upper = np.full(n_rows, np.nan)
    for i, values in enumerate(rows):
        finite = np.isfinite(values)
        if not finite.any():
            continue
        j = int(np.argmax(np.where(finite, values, -np.inf)))
        active = finite & (values >= np.nanmax(values[finite]) - THEORY_LEVEL_TOL)
        lo, hi = _contiguous_run(active, j)
        lower[i], upper[i] = z[lo], z[hi]
    return lower, upper, "level set from F_all/F profile"


def _theory_endpoints(data: np.lib.npyio.NpzFile, n_rows: int, order: np.ndarray,
                      source: Path) -> tuple[np.ndarray, np.ndarray, str]:
    """Resolve direct endpoints first, then documented compatible fallbacks."""
    lo_raw = _pick(data, ("edge_lo", "edge_lower", "maximizer_set_lower",
                          "max_set_lo", "lower"), required=False, source=source)
    hi_raw = _pick(data, ("edge_hi", "edge_upper", "maximizer_set_upper",
                          "max_set_hi", "upper"), required=False, source=source)
    if lo_raw is not None and hi_raw is not None:
        lo = _ordered_vector(lo_raw, n_rows, order, "edge_lo", dtype=float)
        hi = _ordered_vector(hi_raw, n_rows, order, "edge_hi", dtype=float)
        return np.minimum(lo, hi), np.maximum(lo, hi), "edge_lo/edge_hi"

    paired = _pick(data, ("maximizer_set", "maximizer_set_edges", "edge_bounds",
                          "w_extent"), required=False, source=source)
    if paired is not None:
        array = np.asarray(paired, dtype=float)
        if array.shape == (n_rows, 2):
            array = array[order]
        elif array.shape == (2, n_rows):
            array = array.T[order]
        else:
            raise DataLayoutError(f"{source}: endpoint pair has unsupported shape {array.shape}")
        return np.min(array, axis=1), np.max(array, axis=1), "paired endpoint array"

    extent_raw = _pick(data, ("center_extent", "maximizer_half_width",
                              "plateau_half_width"), required=False, source=source)
    if extent_raw is not None:
        extent = np.abs(_ordered_vector(extent_raw, n_rows, order, "center_extent", dtype=float))
        return -extent, extent, "symmetric center_extent"
    return _endpoints_from_profiles(data, n_rows, order, source)


def load_ninf(data_dir: Path, *, allow_partial: bool) -> TheorySeries:
    """Load the n-infinity maximizer set; never reduce it to a raw argmax."""
    ninf_dir = data_dir / "ninf"
    source = _final_or_partial(ninf_dir, "ninf_joint_z3_wi_001_10.npz",
                               allow_partial=allow_partial)
    with np.load(source, allow_pickle=False) as data:
        wi, beta, order = _coordinates(data, source)
        n_rows = len(wi)
        complete = _ordered_vector(
            _pick(data, ("completed", "complete", "is_complete"), required=False, source=source),
            n_rows, order, "completed", dtype=bool, default=True,
        )
        success = _ordered_vector(
            _pick(data, ("center_success", "success_flags", "success"),
                  required=False, source=source),
            n_rows, order, "center_success", dtype=bool, default=True,
        )
        lower, upper, endpoint_source = _theory_endpoints(data, n_rows, order, source)
        keys = tuple(sorted(data.files))
    lower = np.where(complete, lower, np.nan)
    upper = np.where(complete, upper, np.nan)
    lower[np.abs(lower) < 1e-14] = 0.0
    upper[np.abs(upper) < 1e-14] = 0.0
    key, label, alpha, marker = NINF_SPEC
    return TheorySeries(
        key=key, label=label, wi=wi, beta=beta, lower=lower, upper=upper,
        complete=complete, success=success, alpha=alpha, marker=marker,
        source=source, source_keys=keys, endpoint_source=endpoint_source,
    )


def _master_grid(series: Sequence[MCSeries | TheorySeries]) -> np.ndarray:
    """Build the ordered union without interpolating any plotted coordinate."""
    return np.unique(np.concatenate([item.wi for item in series]))


def _onto_grid(master: np.ndarray, wi: np.ndarray, values: np.ndarray) -> np.ndarray:
    output = np.full(len(master), np.nan, dtype=float)
    index = np.searchsorted(master, wi)
    if np.any(index >= len(master)) or not np.allclose(master[index], wi, rtol=0.0, atol=2e-12):
        raise DataLayoutError("A series Wi value is missing from the common export grid")
    output[index] = values
    return output


def _onto_grid_bool(master: np.ndarray, wi: np.ndarray, values: np.ndarray) -> np.ndarray:
    output = np.full(len(master), np.nan, dtype=float)
    index = np.searchsorted(master, wi)
    if np.any(index >= len(master)) or not np.allclose(master[index], wi, rtol=0.0, atol=2e-12):
        raise DataLayoutError("A series Wi value is missing from the common export grid")
    output[index] = np.asarray(values, dtype=float)
    return output


def atomic_csv(path: Path, columns: Sequence[np.ndarray], names: Sequence[str]) -> None:
    """Write CSV atomically, so a reader never sees a half-written plot table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            np.savetxt(handle, np.column_stack(columns), delimiter=",",
                       header=",".join(names), comments="", fmt="%.10g")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_plot_data(series: Sequence[MCSeries | TheorySeries], output_dir: Path) -> Path:
    """Export the exact branch coordinates and diagnostics used in the plot."""
    master = _master_grid(series)
    columns: list[np.ndarray] = [master, master * WI_TO_BETA]
    names = ["Wi", "beta"]
    for item in series:
        if isinstance(item, MCSeries):
            columns.extend([
                _onto_grid(master, item.wi, item.lower),
                _onto_grid(master, item.wi, item.upper),
                _onto_grid(master, item.wi, item.paired_ess_negative),
                _onto_grid(master, item.wi, item.paired_ess_positive),
                _onto_grid_bool(master, item.wi, item.reliable),
                _onto_grid_bool(master, item.wi, item.complete),
                _onto_grid_bool(master, item.wi, item.argmax_at_boundary),
                _onto_grid_bool(master, item.wi, item.quality_pass),
                _onto_grid_bool(master, item.wi, item.profile_stable),
            ])
            names.extend([
                f"z3_argmax_negative_{item.key}",
                f"z3_argmax_positive_{item.key}",
                f"paired_ess_at_negative_argmax_{item.key}",
                f"paired_ess_at_positive_argmax_{item.key}",
                f"paired_ess_ge_{int(ESS_FLOOR)}_{item.key}",
                f"completed_{item.key}",
                f"argmax_at_boundary_{item.key}",
                f"quality_pass_{item.key}",
                f"profile_stable_{item.key}",
            ])
        else:
            columns.extend([
                _onto_grid(master, item.wi, item.lower),
                _onto_grid(master, item.wi, item.upper),
                _onto_grid_bool(master, item.wi, item.complete),
                _onto_grid_bool(master, item.wi, item.success),
            ])
            names.extend([
                "z3_maximizer_set_lower_ninf",
                "z3_maximizer_set_upper_ninf",
                "completed_ninf",
                "center_success_ninf",
            ])
    output = output_dir / "maximizer_vs_wi_001_10_data.csv"
    atomic_csv(output, columns, names)
    return output


def draw(series: Sequence[MCSeries | TheorySeries], output_dir: Path) -> Path:
    """Render an exact-size log-Wi comparison using a single base colour."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for item in series:
        style = dict(color=BASE_COLOR, alpha=item.alpha, lw=2.4,
                     marker=item.marker, ms=5.0, markevery=12,
                     markerfacecolor=BASE_COLOR, markeredgecolor=BASE_COLOR)
        if isinstance(item, TheorySeries):
            ax.fill_between(item.wi, item.lower, item.upper, color=BASE_COLOR,
                            alpha=0.12, linewidth=0, zorder=1)
        ax.plot(item.wi, item.upper, "-", label=item.label, **style)
        ax.plot(item.wi, item.lower, "-", **style)

    finite = np.concatenate([
        np.concatenate([item.lower[np.isfinite(item.lower)], item.upper[np.isfinite(item.upper)]])
        for item in series
    ])
    if len(finite) == 0:
        raise DataLayoutError("No finite maximizer coordinates remain after quality masking")
    y_min, y_max = float(np.min(finite)), float(np.max(finite))
    y_pad = max(0.02, 0.08 * (y_max - y_min))
    ax.axhline(0.0, color="k", ls=":", lw=0.8)
    ax.set_xscale("log")
    ax.set_xlim(0.02, 10.0)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_xlabel(r"$Wi=\beta\,4321/45000$", fontsize=13)
    ax.set_ylabel(r"maximizer location / set edge, $z_3^*$", fontsize=13)
    ax.set_title(r"Joint $(z_2=0,z_3)$ maximizer vs $Wi$", fontsize=13)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=10, loc="upper left", framealpha=0.92)
    fig.tight_layout()
    output = output_dir / "maximizer_vs_wi_001_10.png"
    # No bbox_inches='tight': the requested 1630x955 raster is intentional.
    fig.savefig(output, dpi=OUTPUT_DPI)
    plt.close(fig)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                        help="aggregate directory (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=HERE,
                        help="PNG/CSV directory (default: this script directory)")
    parser.add_argument("--allow-partial", action="store_true",
                        help="fall back to *_partial.npz and mask unfinished rows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mc = [load_mc(*spec, data_dir, allow_partial=args.allow_partial) for spec in MC_SPECS]
    ninf = load_ninf(data_dir, allow_partial=args.allow_partial)
    series: list[MCSeries | TheorySeries] = [*mc, ninf]
    csv = write_plot_data(series, output_dir)
    figure = draw(series, output_dir)
    metadata = {
        "Wi_range": [0.02, 10.0],
        "beta_per_Wi": WI_TO_BETA,
        "ess_floor": ESS_FLOOR,
        "figure_pixels": list(TARGET_PIXELS),
        "series": [
            {
                "key": item.key,
                "source": str(item.source),
                "completed_frames": int(np.count_nonzero(item.complete)),
                "total_frames": int(len(item.wi)),
                "finite_lower_frames": int(np.count_nonzero(np.isfinite(item.lower))),
                "finite_upper_frames": int(np.count_nonzero(np.isfinite(item.upper))),
                **({"endpoint_source": item.endpoint_source}
                   if isinstance(item, TheorySeries) else {}),
            }
            for item in series
        ],
    }
    atomic_json(output_dir / "maximizer_vs_wi_001_10_metadata.json", metadata)
    print(f"  -> {figure}")
    print(f"  -> {csv}")


if __name__ == "__main__":
    main()
