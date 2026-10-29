"""Resumable tilted-MC sweep for the joint ``(z2=0, z3)`` problem.

This is intentionally independent of the historical beta sweeps.  It samples
the exact 340-point Wi grid shared by :mod:`common`, writes one atomic result
per ``(n, Wi)`` point, and only builds its von-Mises proposals from the stored
continuum profiles in ``data/ninf/checkpoints``.  In particular, this program
never imports or re-solves the old continuum variational script.

Examples
--------
Run the seven inexpensive diagnostic locations once their n-infinity frames
exist::

    python -u run_mc_tilted_wi_001_10.py --n 8 --mode preflight

Run the full production grid (resumes atomically after interruption)::

    python -u run_mc_tilted_wi_001_10.py --n 8 --mode production --workers 4

The estimator is the multiple-importance-sampling version of
``../../MC/run_mc_joint_tilted.py``.  Six continuum anchor slope profiles are
interpolated to the finite ``n`` increments, converted to von-Mises tilts, and
all cyclic shifts and their x-reflections (covering z3 < 0) are mixed with the
balance heuristic.  The result is then
symmetrized at the *sample-weight* level, yielding a paired ESS diagnostic for
the even-in-``z3`` target rather than relying on a visually symmetric curve.
"""
from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import os
import time
import zipfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import brentq
from scipy.special import i0e, i1e

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover - optional performance dependency
    threadpool_limits = None

from common import (  # shared grid and atomic helpers; do not duplicate them
    ANCHOR_Z3,
    BETA_PER_WI,
    atomic_savez,
    atomic_write_text,
    beta_from_wi,
    config_hash,
    make_wi_grid,
)


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
NINF_DIR = DATA_DIR / "ninf"
STATE_DIR = HERE / "state"

CONFIG_VERSION = 3
ESTIMATOR = "tilted_vonmises_shift_mirror_mixture_paired_symmetry"
Z2_FIXED = 0.0
Z3_GRID = np.linspace(-0.49, 0.49, 99, dtype=float)
SIGMA_Z2 = 0.02
SIGMA_Z3 = 0.02
N_ANCHORS = 6
ESS_FLOOR = 300.0
PROPOSAL_SLOPE_CAP = 0.999
STABILITY_TOL_PER_N = 0.05
TWO_PI = 2.0 * np.pi
LOG_TWO = float(np.log(2.0))
BASE_SEED = 913_731

# The requested lightweight probes: endpoint, plateau onset and tail.  They
# lie exactly on the 340-point grid and deliberately include both sides of the
# continuum transition beta=4*pi^2 (Wi ~= 3.7908).
PREFLIGHT_WI = np.array([0.02, 3.78, 3.80, 4.00, 6.00, 8.00, 10.00])


class MissingNinfAnchorError(RuntimeError):
    """Raised when the supervisor has not produced a required n-infinity frame."""


def _scalar(value: Any) -> Any:
    """Convert a zero-dimensional NPZ value to a Python scalar."""
    array = np.asarray(value)
    return array.item() if array.ndim == 0 else value


def _as_text(value: Any) -> str:
    value = _scalar(value)
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _sha256_arrays(*arrays: np.ndarray, extra: str = "") -> str:
    digest = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(np.asarray(array))
        digest.update(str(a.dtype).encode("ascii"))
        digest.update(np.asarray(a.shape, dtype=np.int64).tobytes())
        digest.update(a.tobytes())
    digest.update(extra.encode("utf-8"))
    return digest.hexdigest()


def _frame_path(index: int, wi: float) -> Path:
    return NINF_DIR / "checkpoints" / f"frame_{index:03d}_wi_{wi:.6f}.npz"


def load_anchor_profiles(index: int, wi: float, beta: float) -> tuple[np.ndarray, str, str]:
    """Load and validate continuum anchor slopes for one target point.

    The continuum runner owns these files.  A missing or malformed frame is a
    dependency issue, not an invitation to silently run a different solver.
    """
    path = _frame_path(index, wi)
    if not path.exists():
        raise MissingNinfAnchorError(
            f"n-infinity anchors are not ready for index={index} Wi={wi:.6f}: {path}"
        )
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "complete", "config_version", "config_hash", "index", "wi", "beta",
                "anchor_z3", "anchor_profiles", "anchor_success", "anchor_residual",
            }
            missing = required.difference(data.files)
            if missing:
                raise MissingNinfAnchorError(f"{path} misses required keys: {sorted(missing)}")
            if not bool(_scalar(data["complete"])):
                raise MissingNinfAnchorError(f"{path} is not marked complete")
            if int(_scalar(data["index"])) != index:
                raise MissingNinfAnchorError(f"{path} has a mismatched index")
            if not np.isclose(float(_scalar(data["wi"])), wi, rtol=0.0, atol=5e-13):
                raise MissingNinfAnchorError(f"{path} has a mismatched Wi")
            if not np.isclose(float(_scalar(data["beta"])), beta, rtol=0.0, atol=5e-11):
                raise MissingNinfAnchorError(f"{path} has a mismatched beta")
            anchor_z3 = np.asarray(data["anchor_z3"], dtype=float)
            profiles = np.asarray(data["anchor_profiles"], dtype=float)
            if anchor_z3.shape != (N_ANCHORS,) or not np.allclose(
                anchor_z3, ANCHOR_Z3, rtol=0.0, atol=1e-12
            ):
                raise MissingNinfAnchorError(f"{path} has incompatible anchor_z3")
            if profiles.ndim != 2 or profiles.shape[0] != N_ANCHORS or profiles.shape[1] < 4:
                raise MissingNinfAnchorError(f"{path} has incompatible anchor_profiles shape")
            if not np.isfinite(profiles).all():
                raise MissingNinfAnchorError(f"{path} has non-finite anchor_profiles")
            success = np.asarray(data["anchor_success"], dtype=bool)
            residual = np.asarray(data["anchor_residual"], dtype=float)
            if success.shape != (N_ANCHORS,) or not np.all(success):
                raise MissingNinfAnchorError(f"{path} has an uncertified anchor profile")
            if residual.shape != (N_ANCHORS,) or not np.isfinite(residual).all() or np.max(residual) > 2.0e-7:
                raise MissingNinfAnchorError(f"{path} has an anchor constraint residual above 2e-7")
            # This provenance becomes part of checkpoint validation.  Keep the
            # producer's fingerprint as human-readable metadata if available.
            producer_hash = _as_text(data["config_hash"]) if "config_hash" in data else ""
            profile_hash = _sha256_arrays(anchor_z3, profiles, extra=producer_hash)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
        raise MissingNinfAnchorError(f"cannot read n-infinity anchors {path}: {exc}") from exc
    return profiles, profile_hash, producer_hash


# ---------------------------------------------------------------------------
# Stable von-Mises proposal construction
# ---------------------------------------------------------------------------

def _mean_resultant_length(lam: float) -> float:
    """``I1(lam)/I0(lam)`` without overflow."""
    x = abs(float(lam))
    if x < 1e-8:
        return x / 2.0
    # scaled Bessel functions remain finite past the ordinary i0 overflow.
    value = float(i1e(x) / i0e(x))
    if not np.isfinite(value):
        # The leading large-x expansion is more than adequate here; proposal
        # tilts are capped below v=1 anyway.
        value = 1.0 - 0.5 / x
    return min(value, 1.0)


def inv_mean_resultant_length(v: float) -> float:
    """Return lambda such that ``I1(lambda)/I0(lambda) = v``.

    Clipping the *proposal* slope is harmless for unbiasedness and prevents a
    nearly saturated continuum grid value from turning into an unusable
    enormous von-Mises concentration.
    """
    v = float(np.clip(v, -PROPOSAL_SLOPE_CAP, PROPOSAL_SLOPE_CAP))
    if abs(v) < 1e-12:
        return 0.0
    sign = 1.0 if v >= 0.0 else -1.0
    target = abs(v)
    lo, hi = 1e-12, 1.0
    while _mean_resultant_length(hi) < target and hi < 1e6:
        hi *= 2.0
    if _mean_resultant_length(hi) < target:
        # This should only be reachable if a platform Bessel implementation
        # rounds the ratio to one unusually early.  Its asymptotic inverse is
        # still a valid high-quality proposal tilt.
        return sign * (0.5 / max(1.0 - target, 1e-12))
    root = brentq(lambda x: _mean_resultant_length(x) - target, lo, hi, xtol=1e-11)
    return sign * float(root)


def _log_i0(values: np.ndarray) -> np.ndarray:
    """Stable log(I0(values)) from scipy's exponentially scaled I0."""
    x = np.abs(np.asarray(values, dtype=float))
    return np.log(i0e(x)) + x


def build_components(n: int, anchor_profiles: np.ndarray) -> np.ndarray:
    """Interpolate continuum slopes and return all anchor/cyclic/mirror components.

    The anchors only sample z3 >= 0.  The target is even in z3: the
    x-reflection a_x -> -a_x maps (z2, z3) to (-z2, -z3) and leaves the energy
    unchanged, so at z2 = 0 it maps the z3 > 0 sector onto z3 < 0.  Each
    component is therefore paired with its reflected copy (lambda -> -lambda).
    Without these copies the cyclic shifts alone leave the z3 < 0 half almost
    uncovered: there the raw ESS fell to ~10, and because the estimator pairs
    w(z3) with w(-z3) the paired ESS collapsed as well.  That removed the
    |z3| >~ 0.4 cells of n = 32 (and one n = 16 cell) for Wi below ~5, the
    "missing points" of the version-2 runs.
    """
    source_t = (np.arange(anchor_profiles.shape[1], dtype=float) + 0.5) / anchor_profiles.shape[1]
    target_t = (np.arange(n, dtype=float) + 0.5) / n
    components: list[np.ndarray] = []
    for profile in anchor_profiles:
        v_n = np.interp(target_t, source_t, profile)
        v_n = np.clip(v_n, -PROPOSAL_SLOPE_CAP, PROPOSAL_SLOPE_CAP)
        lam = np.array([inv_mean_resultant_length(value) for value in v_n], dtype=float)
        for shift in range(n):
            components.append(np.roll(lam, shift))
            components.append(-np.roll(lam, shift))
    # A zero profile or a periodic profile can make shifts duplicate.  Removing
    # them keeps the mixture normalization and memory cost correct.
    return np.unique(np.round(np.asarray(components), 12), axis=0)


# ---------------------------------------------------------------------------
# Worker estimator
# ---------------------------------------------------------------------------

def _update_log_moments(
    run_max: np.ndarray,
    run_sum: np.ndarray,
    run_sq: np.ndarray,
    log_weights: np.ndarray,
) -> None:
    """Merge a batch into stable sums of weights and squared weights."""
    batch_max = np.max(log_weights, axis=0)
    finite = np.isfinite(batch_max)
    if not np.any(finite):
        return
    scaled = np.exp(log_weights[:, finite] - batch_max[None, finite])
    new_max = np.maximum(run_max[finite], batch_max[finite])
    old_scale = np.exp(run_max[finite] - new_max)
    new_scale = np.exp(batch_max[finite] - new_max)
    run_sum[finite] = run_sum[finite] * old_scale + scaled.sum(axis=0) * new_scale
    run_sq[finite] = (
        run_sq[finite] * old_scale * old_scale
        + (scaled * scaled).sum(axis=0) * new_scale * new_scale
    )
    run_max[finite] = new_max


def _worker_chunk(args: tuple[Any, ...]) -> tuple[np.ndarray, ...]:
    """One independent worker stream; results are sufficient for resume diagnostics."""
    (
        n,
        m_chunk,
        seed,
        f_value,
        z3_values,
        sigma_z2,
        sigma_z3,
        batch,
        components,
    ) = args
    if threadpool_limits is not None:
        threadpool_limits(limits=1)

    rng = np.random.default_rng(seed)
    components = np.asarray(components, dtype=float)
    n_components = components.shape[0]
    log_i0_sum = _log_i0(components).sum(axis=1)
    means = np.where(components >= 0.0, 0.0, np.pi)
    kappas = np.abs(components)
    n_z = len(z3_values)

    raw_max = np.full(n_z, -np.inf)
    raw_sum = np.zeros(n_z)
    raw_sq = np.zeros(n_z)
    pair_max = np.full(n_z, -np.inf)
    pair_sum = np.zeros(n_z)
    pair_sq = np.zeros(n_z)

    completed = 0
    while completed < m_chunk:
        b = min(batch, m_chunk - completed)
        component_index = rng.integers(0, n_components, size=b)
        theta = rng.vonmises(means[component_index], kappas[component_index])
        ax = np.cos(theta)
        ay = np.sin(theta)

        rx = np.concatenate((np.zeros((b, 1)), np.cumsum(ax, axis=1)), axis=1)
        ry = np.concatenate((np.zeros((b, 1)), np.cumsum(ay, axis=1)), axis=1)
        dx = rx - rx.mean(axis=1, keepdims=True)
        dy = ry - ry.mean(axis=1, keepdims=True)
        energy = (dx * dx - dy * dy).sum(axis=1)
        z2 = rx[:, -1] / n
        z3 = z2 - 2.0 * rx.mean(axis=1) / n

        # q / mu for a uniform mixture of tilted product measures.  log_lr is
        # therefore log(mu / q), exactly as in the historical tilted runner.
        exponent = ax @ components.T - log_i0_sum[None, :]
        exponent_max = exponent.max(axis=1, keepdims=True)
        log_q_over_mu = (
            exponent_max[:, 0]
            + np.log(np.exp(exponent - exponent_max).sum(axis=1))
            - np.log(n_components)
        )
        base = (
            f_value * energy
            - 0.5 * ((z2 - Z2_FIXED) / sigma_z2) ** 2
            - log_q_over_mu
        )
        log_weights = base[:, None] - 0.5 * ((z3[:, None] - z3_values[None, :]) / sigma_z3) ** 2
        _update_log_moments(raw_max, raw_sum, raw_sq, log_weights)

        # Pair z3 and -z3 before computing ESS.  This is an unbiased estimate
        # of the symmetric target with ordinary ESS semantics for its actual
        # sample weights, not merely min(ESS(+z), ESS(-z)).
        paired_log_weights = np.logaddexp(log_weights, log_weights[:, ::-1]) - LOG_TWO
        _update_log_moments(pair_max, pair_sum, pair_sq, paired_log_weights)
        completed += b

    return raw_max, raw_sum, raw_sq, pair_max, pair_sum, pair_sq, np.array(m_chunk)


def _combine_moments(
    maxima: np.ndarray,
    sums: np.ndarray,
    squares: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Combine worker log-moment accumulators into log mean and ESS."""
    global_max = np.max(maxima, axis=0)
    finite = np.isfinite(global_max)
    total = np.zeros(global_max.shape, dtype=float)
    total_sq = np.zeros(global_max.shape, dtype=float)
    if np.any(finite):
        scale = np.exp(maxima[:, finite] - global_max[None, finite])
        total[finite] = (sums[:, finite] * scale).sum(axis=0)
        total_sq[finite] = (squares[:, finite] * scale * scale).sum(axis=0)
    ess = np.zeros(global_max.shape, dtype=float)
    usable = finite & (total > 0.0) & (total_sq > 0.0)
    ess[usable] = total[usable] ** 2 / total_sq[usable]
    return global_max, total, ess


def _logz_from_moments(maxima: np.ndarray, sums: np.ndarray, sample_count: int) -> np.ndarray:
    output = np.full(maxima.shape, -np.inf, dtype=float)
    usable = np.isfinite(maxima) & (sums > 0.0)
    if np.any(usable):
        output[usable] = (
            maxima[usable]
            + np.log(sums[usable] / sample_count)
            - np.log(TWO_PI)
            - np.log(SIGMA_Z2)
            - np.log(SIGMA_Z3)
        )
    return output


def _split_profile_disagreement(
    pair_max: np.ndarray,
    pair_sum: np.ndarray,
    pair_sq: np.ndarray,
    m_per_worker: int,
    n: int,
) -> tuple[float, bool]:
    """Compare two independent worker groups after removing their offsets."""
    workers = pair_max.shape[0]
    if workers < 2:
        return float("inf"), False
    split = workers // 2
    groups = (np.arange(split), np.arange(split, workers))
    profiles: list[np.ndarray] = []
    for ids in groups:
        mx, total, ess = _combine_moments(pair_max[ids], pair_sum[ids], pair_sq[ids])
        profile = _logz_from_moments(mx, total, m_per_worker * len(ids))
        # Each split contains only roughly half the full estimator's ESS.  Use
        # a lower screen here; the full paired ESS still has the strict 300
        # threshold in ``_point_quality`` below.
        profile[ess < max(20.0, ESS_FLOOR / 4.0)] = np.nan
        profiles.append(profile)
    midpoint = len(Z3_GRID) // 2
    a, b = profiles
    valid = np.isfinite(a) & np.isfinite(b) & np.isfinite(a[midpoint]) & np.isfinite(b[midpoint])
    if not np.any(valid):
        return float("inf"), False
    delta = (a - a[midpoint]) / n - (b - b[midpoint]) / n
    disagreement = float(np.max(np.abs(delta[valid])))
    return disagreement, bool(disagreement <= STABILITY_TOL_PER_N)


def tilted_logz(
    *,
    n: int,
    beta: float,
    components: np.ndarray,
    m_requested: int,
    batch: int,
    workers: int,
    seed: int,
) -> dict[str, Any]:
    """Evaluate a symmetric 99-point profile and return raw + paired diagnostics."""
    if workers < 2:
        raise ValueError("at least two workers are required for the stability diagnostic")
    m_per_worker = (m_requested + workers - 1) // workers
    actual_m = m_per_worker * workers
    f_value = beta / (n * n)
    tasks = [
        (
            n,
            m_per_worker,
            seed + 7_919 * worker,
            f_value,
            Z3_GRID,
            SIGMA_Z2,
            SIGMA_Z3,
            batch,
            components,
        )
        for worker in range(workers)
    ]
    started = time.time()
    if workers == 1:  # retained for direct API use; CLI deliberately rejects it
        results = [_worker_chunk(tasks[0])]
    else:
        with mp.get_context("fork").Pool(workers) as pool:
            # Preserve the deterministic seed order: the split-stream
            # stability diagnostic below must not depend on worker completion
            # timing.
            results = pool.map(_worker_chunk, tasks)

    raw_max = np.stack([item[0] for item in results])
    raw_sum = np.stack([item[1] for item in results])
    raw_sq = np.stack([item[2] for item in results])
    pair_max = np.stack([item[3] for item in results])
    pair_sum = np.stack([item[4] for item in results])
    pair_sq = np.stack([item[5] for item in results])

    raw_global_max, raw_total, raw_ess = _combine_moments(raw_max, raw_sum, raw_sq)
    pair_global_max, pair_total, paired_ess = _combine_moments(pair_max, pair_sum, pair_sq)
    raw_log_z = _logz_from_moments(raw_global_max, raw_total, actual_m)
    paired_log_z = _logz_from_moments(pair_global_max, pair_total, actual_m)

    # Worker-level SE is only a rough diagnostic, but it catches split-stream
    # disagreement independently of ESS.  Compute it for the paired profile.
    worker_log_z = np.stack([
        _logz_from_moments(pair_max[k], pair_sum[k], m_per_worker)
        for k in range(workers)
    ])
    se = np.nanstd(np.where(np.isfinite(worker_log_z), worker_log_z, np.nan), axis=0, ddof=1)
    se /= np.sqrt(workers)
    disagreement, profile_stable = _split_profile_disagreement(
        pair_max, pair_sum, pair_sq, m_per_worker, n
    )
    return {
        "mc_logZ": paired_log_z,
        "mc_logZ_raw": raw_log_z,
        "ess": raw_ess,
        "paired_ess": paired_ess,
        "se": se,
        "split_profile_disagreement_per_n": disagreement,
        "profile_stable": profile_stable,
        "M_actual": actual_m,
        "elapsed_seconds": time.time() - started,
    }


# ---------------------------------------------------------------------------
# Point diagnostics, checkpoint validity, and aggregation
# ---------------------------------------------------------------------------

def _mode_defaults(mode: str) -> tuple[int, int]:
    if mode == "preflight":
        return 2_000_000, 20_000_000
    return 20_000_000, 100_000_000


def _selected_indices(mode: str, explicit_indices: list[int] | None, explicit_wi: list[float] | None) -> np.ndarray:
    wi = make_wi_grid()
    if explicit_indices is not None:
        result = np.array(sorted(set(explicit_indices)), dtype=int)
        if result.size == 0 or result.min() < 0 or result.max() >= len(wi):
            raise ValueError(f"--indices must be within 0..{len(wi) - 1}")
        return result
    if explicit_wi is not None:
        ids = []
        for value in explicit_wi:
            index = int(np.argmin(np.abs(wi - value)))
            if not np.isclose(wi[index], value, rtol=0.0, atol=5e-7):
                raise ValueError(f"requested Wi={value:g} is not on the fixed grid")
            ids.append(index)
        return np.array(sorted(set(ids)), dtype=int)
    if mode == "preflight":
        return np.array([int(np.argmin(np.abs(wi - value))) for value in PREFLIGHT_WI], dtype=int)
    return np.arange(len(wi), dtype=int)


def _run_config(mode: str, *, n: int, batch: int, workers: int, base_m: int, max_m: int) -> dict[str, Any]:
    """Only fields that change estimator/resume semantics belong here."""
    return {
        "config_version": CONFIG_VERSION,
        "mode": mode,
        "n": n,
        "z2_fixed": Z2_FIXED,
        "z3_grid": [float(x) for x in Z3_GRID],
        "sigma_z2": SIGMA_Z2,
        "sigma_z3": SIGMA_Z3,
        "anchor_z3": [float(x) for x in ANCHOR_Z3],
        "proposal_slope_cap": PROPOSAL_SLOPE_CAP,
        "estimator": ESTIMATOR,
        "batch": batch,
        "workers": workers,
        "M_base": base_m,
        "M_max": max_m,
        "ess_floor": ESS_FLOOR,
        "stability_tol_per_n": STABILITY_TOL_PER_N,
    }


def _point_path(mode: str, n: int, index: int, wi: float) -> Path:
    return HERE / "mc_points" / mode / f"n{n}" / f"point_{index:03d}_wi_{wi:.6f}.npz"


def _scalar_bool(data: Any, key: str) -> bool:
    return bool(_scalar(data[key]))


def point_is_valid(
    path: Path,
    *,
    index: int,
    wi: float,
    beta: float,
    n: int,
    runner_hash: str,
    profile_hash: str,
) -> bool:
    """Strictly reject partial, stale, or incompatible point files."""
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            required = {
                "complete", "config_version", "runner_config_hash", "ninf_profile_hash",
                "index", "wi", "beta", "n", "z3_grid", "mc_logZ", "mc_logZ_raw",
                "ess", "paired_ess", "se", "M_requested", "M_actual", "quality_pass",
                "adaptive_exhausted", "argmax_at_boundary", "profile_stable",
            }
            if required.difference(data.files):
                return False
            return bool(
                _scalar_bool(data, "complete")
                and int(_scalar(data["config_version"])) == CONFIG_VERSION
                and _as_text(data["runner_config_hash"]) == runner_hash
                and _as_text(data["ninf_profile_hash"]) == profile_hash
                and int(_scalar(data["index"])) == index
                and int(_scalar(data["n"])) == n
                and np.isclose(float(_scalar(data["wi"])), wi, rtol=0.0, atol=5e-13)
                and np.isclose(float(_scalar(data["beta"])), beta, rtol=0.0, atol=5e-11)
                and np.array_equal(np.asarray(data["z3_grid"]), Z3_GRID)
                and np.asarray(data["mc_logZ"]).shape == Z3_GRID.shape
                and np.asarray(data["paired_ess"]).shape == Z3_GRID.shape
                and np.isfinite(np.asarray(data["mc_logZ"])).all()
                and np.isfinite(np.asarray(data["se"])).all()
            )
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return False


def _point_quality(result: dict[str, Any]) -> tuple[bool, bool, float, bool, str, float, float]:
    """Return quality decision with individual evidence preserved in the NPZ."""
    profile = np.asarray(result["mc_logZ"], dtype=float)
    paired_ess = np.asarray(result["paired_ess"], dtype=float)
    midpoint = len(Z3_GRID) // 2
    positive = Z3_GRID >= 0.0
    negative = Z3_GRID <= 0.0
    relative = (profile - profile[midpoint])
    z_star_pos = float(Z3_GRID[positive][np.argmax(relative[positive])])
    z_star_neg = float(Z3_GRID[negative][np.argmax(relative[negative])])
    at_boundary = bool(
        np.isclose(z_star_pos, Z3_GRID[-1]) or np.isclose(z_star_neg, Z3_GRID[0])
    )
    paired_ess_min = float(np.min(paired_ess))
    paired_ess_center = float(paired_ess[midpoint])
    stable = bool(result["profile_stable"])
    reasons = []
    if paired_ess_min < ESS_FLOOR:
        reasons.append(f"paired_ess_min<{ESS_FLOOR:g}")
    if at_boundary:
        reasons.append("argmax_at_z3_boundary")
    if not stable:
        reasons.append("split_profile_unstable")
    return (
        not reasons,
        at_boundary,
        paired_ess_min,
        stable,
        "ok" if not reasons else ";".join(reasons),
        z_star_neg,
        z_star_pos,
    )


def _read_point_summary(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "mc_logZ": np.asarray(data["mc_logZ"], dtype=float),
            "mc_logZ_raw": np.asarray(data["mc_logZ_raw"], dtype=float),
            "ess": np.asarray(data["ess"], dtype=float),
            "paired_ess": np.asarray(data["paired_ess"], dtype=float),
            "se": np.asarray(data["se"], dtype=float),
            "M_requested": int(_scalar(data["M_requested"])),
            "M_actual": int(_scalar(data["M_actual"])),
            "quality_pass": _scalar_bool(data, "quality_pass"),
            "adaptive_exhausted": _scalar_bool(data, "adaptive_exhausted"),
            "argmax_at_boundary": _scalar_bool(data, "argmax_at_boundary"),
            "profile_stable": _scalar_bool(data, "profile_stable"),
            "paired_ess_min": float(_scalar(data["paired_ess_min"])),
            "paired_ess_center": float(_scalar(data["paired_ess_center"])),
            "split_profile_disagreement_per_n": float(_scalar(data["split_profile_disagreement_per_n"])),
            "quality_reason": _as_text(data["quality_reason"]),
            "rerun_count": int(_scalar(data["rerun_count"])),
            "z_star_neg": float(_scalar(data["z_star_neg"])),
            "z_star_pos": float(_scalar(data["z_star_pos"])),
            "ninf_profile_hash": _as_text(data["ninf_profile_hash"]),
            "ninf_source_config_hash": _as_text(data["ninf_source_config_hash"]),
            "component_count": int(_scalar(data["component_count"])),
        }


def _aggregate_paths(mode: str, n: int) -> tuple[Path, Path]:
    if mode == "production":
        root = DATA_DIR
    else:
        root = DATA_DIR / "preflight"
    stem = f"mc_n{n}_joint_z3_wi_001_10"
    return root / f"{stem}_partial.npz", root / f"{stem}.npz"


def assemble(
    *,
    mode: str,
    n: int,
    selected: np.ndarray,
    runner_hash: str,
) -> tuple[int, int]:
    """Write mode-local partial aggregate and final aggregate when complete."""
    wi = make_wi_grid()
    beta = np.asarray(beta_from_wi(wi), dtype=float)
    rows, cols = len(wi), len(Z3_GRID)
    logz = np.full((rows, cols), np.nan)
    raw_logz = np.full_like(logz, np.nan)
    ess = np.full_like(logz, np.nan)
    paired_ess = np.full_like(logz, np.nan)
    se = np.full_like(logz, np.nan)
    completed = np.zeros(rows, dtype=bool)
    quality_pass = np.zeros(rows, dtype=bool)
    adaptive_exhausted = np.zeros(rows, dtype=bool)
    boundary = np.zeros(rows, dtype=bool)
    stable = np.zeros(rows, dtype=bool)
    m_requested = np.zeros(rows, dtype=np.int64)
    m_actual = np.zeros(rows, dtype=np.int64)
    rerun_count = np.zeros(rows, dtype=np.int16)
    paired_ess_min = np.full(rows, np.nan)
    paired_ess_center = np.full(rows, np.nan)
    disagreement = np.full(rows, np.nan)
    z_star_neg = np.full(rows, np.nan)
    z_star_pos = np.full(rows, np.nan)
    component_count = np.zeros(rows, dtype=np.int16)
    quality_reason = np.full(rows, "", dtype="U128")
    profile_hash = np.full(rows, "", dtype="U64")
    ninf_config_hash = np.full(rows, "", dtype="U64")

    for index in selected:
        point = _point_path(mode, n, int(index), float(wi[index]))
        if not point.exists():
            continue
        try:
            row = _read_point_summary(point)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            continue
        # Assemble only point files made by this exact runner configuration.
        try:
            with np.load(point, allow_pickle=False) as check:
                if (
                    not _scalar_bool(check, "complete")
                    or _as_text(check["runner_config_hash"]) != runner_hash
                    or int(_scalar(check["index"])) != int(index)
                ):
                    continue
        except (OSError, ValueError, KeyError, zipfile.BadZipFile):
            continue
        logz[index] = row["mc_logZ"]
        raw_logz[index] = row["mc_logZ_raw"]
        ess[index] = row["ess"]
        paired_ess[index] = row["paired_ess"]
        se[index] = row["se"]
        completed[index] = True
        quality_pass[index] = row["quality_pass"]
        adaptive_exhausted[index] = row["adaptive_exhausted"]
        boundary[index] = row["argmax_at_boundary"]
        stable[index] = row["profile_stable"]
        m_requested[index] = row["M_requested"]
        m_actual[index] = row["M_actual"]
        rerun_count[index] = row["rerun_count"]
        paired_ess_min[index] = row["paired_ess_min"]
        paired_ess_center[index] = row["paired_ess_center"]
        disagreement[index] = row["split_profile_disagreement_per_n"]
        z_star_neg[index] = row["z_star_neg"]
        z_star_pos[index] = row["z_star_pos"]
        component_count[index] = row["component_count"]
        quality_reason[index] = row["quality_reason"]
        profile_hash[index] = row["ninf_profile_hash"]
        ninf_config_hash[index] = row["ninf_source_config_hash"]

    intended = np.zeros(rows, dtype=bool)
    intended[selected] = True
    metadata = dict(
        config_version=np.array(CONFIG_VERSION),
        runner_config_hash=np.array(runner_hash),
        estimator=np.array(ESTIMATOR),
        mode=np.array(mode),
        n=np.array(n),
        wi_values=wi,
        beta_values=beta,
        f_values=beta / n**2,
        z3_grid=Z3_GRID,
        z_grid=Z3_GRID,  # a convenience alias for earlier plotting scripts
        mc_logZ=logz,
        mc_logZ_raw=raw_logz,
        ess=ess,
        paired_ess=paired_ess,
        se=se,
        completed=completed,
        intended=intended,
        quality_pass=quality_pass,
        adaptive_exhausted=adaptive_exhausted,
        argmax_at_boundary=boundary,
        profile_stable=stable,
        M_requested=m_requested,
        M_actual=m_actual,
        rerun_count=rerun_count,
        paired_ess_min=paired_ess_min,
        paired_ess_center=paired_ess_center,
        split_profile_disagreement_per_n=disagreement,
        z_star_neg=z_star_neg,
        z_star_pos=z_star_pos,
        component_count=component_count,
        quality_reason=quality_reason,
        ninf_profile_hash=profile_hash,
        ninf_source_config_hash=ninf_config_hash,
        sigma_z2=np.array(SIGMA_Z2),
        sigma_z3=np.array(SIGMA_Z3),
        anchor_z3=ANCHOR_Z3,
        ess_floor=np.array(ESS_FLOOR),
        stability_tol_per_n=np.array(STABILITY_TOL_PER_N),
    )
    partial, final = _aggregate_paths(mode, n)
    atomic_savez(partial, **metadata)

    lines = [
        "index\tWi\tbeta\tstatus\tquality\tM_requested\tpaired_ESS_min\tboundary\tstable\tdisagreement_per_n\treason",
    ]
    for index in selected:
        if completed[index]:
            status = "done" if quality_pass[index] else "exhausted" if adaptive_exhausted[index] else "needs_rerun"
            lines.append(
                f"{index}\t{wi[index]:.6f}\t{beta[index]:.12g}\t{status}\t{int(quality_pass[index])}\t"
                f"{m_requested[index]}\t{paired_ess_min[index]:.6g}\t{int(boundary[index])}\t"
                f"{int(stable[index])}\t{disagreement[index]:.6g}\t{quality_reason[index]}"
            )
        else:
            lines.append(f"{index}\t{wi[index]:.6f}\t{beta[index]:.12g}\tpending\t\t\t\t\t\t\t")
    atomic_write_text(STATE_DIR / f"mc_n{n}_{mode}_progress.tsv", "\n".join(lines) + "\n")

    done = int(completed[selected].sum())
    if done == len(selected):
        atomic_savez(final, **metadata)
    return done, len(selected)


# ---------------------------------------------------------------------------
# Process lifecycle and main loop
# ---------------------------------------------------------------------------

def _lock_path(mode: str, n: int) -> Path:
    return STATE_DIR / f"mc_n{n}_{mode}.lock"


def _process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_lock(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            old_pid = int(path.read_text(encoding="utf-8").split()[0])
        except (OSError, ValueError, IndexError):
            old_pid = -1
        if _process_is_alive(old_pid):
            raise RuntimeError(f"MC lock is held by PID {old_pid}: {path}")
        path.unlink()
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        handle.write(f"{os.getpid()} {time.time():.6f}\n")
        handle.flush()
        os.fsync(handle.fileno())


def _next_m(previous: int, maximum: int) -> int:
    """The requested 2e7 -> 1e8 production escalation, generalized safely."""
    if previous >= maximum:
        return previous
    return min(maximum, max(previous * 5, previous + 1))


def _existing_attempt(path: Path) -> tuple[int, int] | None:
    """Return (M_requested, rerun_count) from a validly readable old point."""
    try:
        with np.load(path, allow_pickle=False) as data:
            return int(_scalar(data["M_requested"])), int(_scalar(data["rerun_count"]))
    except (OSError, ValueError, KeyError, zipfile.BadZipFile):
        return None


def _evaluate_and_checkpoint(
    *,
    point: Path,
    mode: str,
    run_config: dict[str, Any],
    runner_hash: str,
    profile_hash: str,
    source_hash: str,
    n: int,
    index: int,
    wi: float,
    beta: float,
    components: np.ndarray,
    m_attempt: int,
    base_m: int,
    max_m: int,
    batch: int,
    workers: int,
    rerun_count: int,
    seed: int,
) -> tuple[bool, bool, float, bool, float]:
    """Run one independent adaptive attempt and atomically publish it.

    Returns ``(quality, exhausted, paired_ess_min, stable, elapsed_seconds)``.
    The caller may immediately invoke this again with five times the sample
    count when the diagnostics request an adaptive rerun.
    """
    result = tilted_logz(
        n=n,
        beta=beta,
        components=components,
        m_requested=m_attempt,
        batch=batch,
        workers=workers,
        seed=seed,
    )
    if not np.isfinite(result["mc_logZ"]).all() or not np.isfinite(result["se"]).all():
        raise FloatingPointError(f"non-finite tilted MC output at Wi={wi:.6f}")
    quality, boundary, ess_min, stable, reason, z_neg, z_pos = _point_quality(result)
    exhausted = bool((not quality) and m_attempt >= max_m)
    midpoint = len(Z3_GRID) // 2
    relative = (np.asarray(result["mc_logZ"]) - result["mc_logZ"][midpoint]) / n
    relative_se = np.sqrt(np.asarray(result["se"]) ** 2 + result["se"][midpoint] ** 2) / n
    atomic_savez(
        point,
        complete=np.array(True),
        config_version=np.array(CONFIG_VERSION),
        runner_config_hash=np.array(runner_hash),
        runner_config_json=np.array(str(run_config)),
        ninf_profile_hash=np.array(profile_hash),
        ninf_source_config_hash=np.array(source_hash),
        estimator=np.array(ESTIMATOR),
        mode=np.array(mode),
        index=np.array(index),
        wi=np.array(wi),
        beta=np.array(beta),
        n=np.array(n),
        z2_fixed=np.array(Z2_FIXED),
        z3_grid=Z3_GRID,
        anchor_z3=ANCHOR_Z3,
        component_count=np.array(len(components)),
        proposal_slope_cap=np.array(PROPOSAL_SLOPE_CAP),
        mc_logZ=np.asarray(result["mc_logZ"]),
        mc_logZ_raw=np.asarray(result["mc_logZ_raw"]),
        log_Z=np.asarray(result["mc_logZ"]),  # compatibility alias
        ess=np.asarray(result["ess"]),
        paired_ess=np.asarray(result["paired_ess"]),
        se=np.asarray(result["se"]),
        relative_log_Z_per_n=relative,
        relative_se_per_n=relative_se,
        paired_ess_min=np.array(ess_min),
        paired_ess_center=np.array(result["paired_ess"][midpoint]),
        z_star_neg=np.array(z_neg),
        z_star_pos=np.array(z_pos),
        argmax_at_boundary=np.array(boundary),
        profile_stable=np.array(stable),
        split_profile_disagreement_per_n=np.array(result["split_profile_disagreement_per_n"]),
        quality_pass=np.array(quality),
        quality_reason=np.array(reason),
        adaptive_exhausted=np.array(exhausted),
        rerun_count=np.array(rerun_count),
        M_requested=np.array(m_attempt),
        M_actual=np.array(result["M_actual"]),
        M_base=np.array(base_m),
        M_max=np.array(max_m),
        batch=np.array(batch),
        n_workers=np.array(workers),
        sigma_z2=np.array(SIGMA_Z2),
        sigma_z3=np.array(SIGMA_Z3),
        ess_floor=np.array(ESS_FLOOR),
        stability_tol_per_n=np.array(STABILITY_TOL_PER_N),
        seed=np.array(seed),
        elapsed_seconds=np.array(result["elapsed_seconds"]),
    )
    return quality, exhausted, ess_min, stable, float(result["elapsed_seconds"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, choices=(8, 16, 32), required=True)
    parser.add_argument("--mode", choices=("preflight", "production"), required=True)
    parser.add_argument("--M", type=int, default=None, help="Baseline samples per point; mode default if omitted.")
    parser.add_argument("--max-M", type=int, default=None, help="Cap for adaptive full reruns; mode default if omitted.")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("MC_WORKERS", "4")))
    parser.add_argument("--batch", type=int, default=int(os.environ.get("MC_BATCH", "20000")))
    parser.add_argument("--indices", type=int, nargs="+", default=None, help="Testing/subset override using fixed-grid indices.")
    parser.add_argument("--wi", type=float, nargs="+", default=None, help="Testing/subset override using exact fixed-grid Wi values.")
    parser.add_argument("--max-points", type=int, default=None, help="Stop after this many newly processed Wi points.")
    parser.add_argument("--retry-exhausted", action="store_true", help="Re-run points that previously reached max-M without quality passing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.indices is not None and args.wi is not None:
        raise ValueError("use only one of --indices and --wi")
    if args.workers < 2:
        raise ValueError("--workers must be at least 2 for split stability diagnostics")
    if args.batch <= 0:
        raise ValueError("--batch must be positive")
    default_m, default_max_m = _mode_defaults(args.mode)
    base_m = default_m if args.M is None else args.M
    max_m = default_max_m if args.max_M is None else args.max_M
    if base_m <= 0 or max_m < base_m:
        raise ValueError("require positive --M and --max-M >= --M")
    selected = _selected_indices(args.mode, args.indices, args.wi)
    wi = make_wi_grid()
    beta_values = np.asarray(beta_from_wi(wi), dtype=float)
    run_config = _run_config(
        args.mode, n=args.n, batch=args.batch, workers=args.workers, base_m=base_m, max_m=max_m
    )
    runner_hash = config_hash(run_config)
    lock = _lock_path(args.mode, args.n)
    acquire_lock(lock)
    status_path = STATE_DIR / f"mc_n{args.n}_{args.mode}.status"
    try:
        atomic_write_text(
            status_path,
            f"running pid={os.getpid()} mode={args.mode} started_unix={time.time():.6f} config={runner_hash}\n",
        )
        print(
            f"tilted MC n={args.n} mode={args.mode}: {len(selected)} fixed Wi points "
            f"({wi[selected[0]]:.6f}..{wi[selected[-1]]:.6f}), M={base_m:,}->{max_m:,}, "
            f"workers={args.workers}, batch={args.batch}",
            flush=True,
        )
        done, total = assemble(mode=args.mode, n=args.n, selected=selected, runner_hash=runner_hash)
        print(f"  resume state: {done}/{total} compatible point checkpoints", flush=True)
        newly_processed = 0

        for index in selected:
            index = int(index)
            point = _point_path(args.mode, args.n, index, float(wi[index]))
            profiles, profile_hash, source_hash = load_anchor_profiles(index, float(wi[index]), float(beta_values[index]))
            valid = point_is_valid(
                point,
                index=index,
                wi=float(wi[index]),
                beta=float(beta_values[index]),
                n=args.n,
                runner_hash=runner_hash,
                profile_hash=profile_hash,
            )
            previous = _existing_attempt(point) if valid else None
            if valid:
                with np.load(point, allow_pickle=False) as prior:
                    pass_quality = _scalar_bool(prior, "quality_pass")
                    exhausted = _scalar_bool(prior, "adaptive_exhausted")
                if pass_quality or (exhausted and not args.retry_exhausted):
                    continue
            if args.max_points is not None and newly_processed >= args.max_points:
                break

            if previous is None:
                m_attempt, rerun_count = base_m, 0
            else:
                m_attempt, rerun_count = _next_m(previous[0], max_m), previous[1] + 1
            components = build_components(args.n, profiles)
            is_new_point = True
            while True:
                # A stale/corrupt file must not steer the random stream.  The
                # seed changes per adaptive attempt but remains deterministic
                # on resume, and every completed attempt is checkpointed.
                seed = (
                    BASE_SEED + 1_000_003 * args.n + 10_007 * index
                    + 104_729 * rerun_count
                )
                print(
                    f"  [{index + 1:03d}/{len(wi)}] Wi={wi[index]:.6f} beta={beta_values[index]:.8g} "
                    f"M={m_attempt:,} attempt={rerun_count} components={len(components)}",
                    flush=True,
                )
                quality, exhausted, ess_min, stable, elapsed = _evaluate_and_checkpoint(
                    point=point,
                    mode=args.mode,
                    run_config=run_config,
                    runner_hash=runner_hash,
                    profile_hash=profile_hash,
                    source_hash=source_hash,
                    n=args.n,
                    index=index,
                    wi=float(wi[index]),
                    beta=float(beta_values[index]),
                    components=components,
                    m_attempt=m_attempt,
                    base_m=base_m,
                    max_m=max_m,
                    batch=args.batch,
                    workers=args.workers,
                    rerun_count=rerun_count,
                    seed=seed,
                )
                if is_new_point:
                    newly_processed += 1
                    is_new_point = False
                done, total = assemble(mode=args.mode, n=args.n, selected=selected, runner_hash=runner_hash)
                print(
                    f"    saved: paired ESS min={ess_min:.1f}, stable={stable}, quality={quality}, "
                    f"exhausted={exhausted}, elapsed={elapsed:.1f}s; complete={done}/{total}",
                    flush=True,
                )
                if quality or exhausted:
                    break
                m_attempt = _next_m(m_attempt, max_m)
                rerun_count += 1

        done, total = assemble(mode=args.mode, n=args.n, selected=selected, runner_hash=runner_hash)
        state = "complete" if done == total else "paused"
        atomic_write_text(status_path, f"{state} pid={os.getpid()} mode={args.mode} points={done}/{total}\n")
        print(f"{state}: {done}/{total} points", flush=True)
    except Exception:
        atomic_write_text(status_path, f"failed pid={os.getpid()} mode={args.mode}\n")
        raise
    finally:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
