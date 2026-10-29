#!/usr/bin/env python3
"""Resumable n -> infinity sweep on the Wi = 0.01 .. 10 grid.

This is deliberately independent of ``Inf_limt/run_ninf_beta_5_12.py``.
The old sweep is adequate in its low-beta window, but its single uniform
initialization gets trapped on the wrong stationary branch above
``beta = 4*pi**2`` and it writes its cache only after the whole sweep.  This
runner instead:

* uses a 141-point Wi grid on Wi = 0.02 .. 10;
* writes one atomically replaced checkpoint per Wi value;
* uses scaled Bessel functions (``i0e`` / ``i1e``) for a stable rate table;
* applies deterministic multistart optimization and records all convergence
  diagnostics; and
* mirrors the positive-|z| solution, so the numerical result is exactly even
  in z.

Normal use (safe to resume after a disconnected background job):

    python -u run_ninf_wi_001_10.py

Useful small checks:

    python run_ninf_wi_001_10.py --index 0 --force --no-aggregate
    python run_ninf_wi_001_10.py --max-frames 1

Outputs live below ``data/ninf/`` next to this script.  A frame is skipped on
restart only when its configuration fingerprint, grid, finite values,
profiles, residuals, and solver-success flags all validate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import traceback
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from scipy.optimize import minimize, root_scalar
from scipy.special import i0e, i1e


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = HERE / "data" / "ninf"

WI_PER_BETA = 4321.0 / 45000.0
BETA_PER_WI = 1.0 / WI_PER_BETA
BETA_NONUNIFORM_ONSET = 4.0 * np.pi**2

SCHEMA_VERSION = "ninf-wi-001-10-v1"
CONSTRAINT_TOL = 2.0e-7
OBJECTIVE_TOL = 1.0e-7


def make_wi_grid() -> np.ndarray:
    """Return the 141-point Wi grid on [0.02, 10], rounded exactly as its reference."""
    # The reference grid below is in the earlier Wi labels.  Wi = beta*4321/45000
    # doubles them with beta unchanged; only Wi <= 10 is kept, plus Wi = 10 itself.
    wi = 2.0 * np.unique(
        np.round(
            np.concatenate(
                [
                    np.array([0.01]),
                    np.arange(0.02, 1.0001, 0.02),
                    np.arange(0.15, 0.5501, 0.005),
                    np.arange(1.05, 2.0001, 0.05),
                    np.geomspace(2.2, 10.0, 15),
                ]
            ),
            6,
        )
    )
    wi = np.unique(np.r_[wi[wi <= 10.0], 10.0])
    if len(wi) != 141 or not np.isclose(wi[0], 0.02) or not np.isclose(wi[-1], 10.0):
        raise RuntimeError("The fixed Wi grid is not the expected 141-point grid.")
    return wi


def make_z_grid(points: int, z_max: float) -> np.ndarray:
    if points < 3 or points % 2 == 0:
        raise ValueError("--z-points must be odd and at least 3 so z=0 is present.")
    if not (0.0 < z_max < 1.0):
        raise ValueError("--z-max must lie strictly between 0 and 1.")
    grid = np.linspace(-z_max, z_max, points)
    if not np.isclose(grid[points // 2], 0.0):
        raise RuntimeError("The z grid does not contain zero.")
    return grid


@dataclass(frozen=True)
class SolverConfig:
    n_steps: int
    z_grid: np.ndarray
    v_table_cap: float
    v_bound: float
    maxiter: int
    retry_maxiter: int
    ftol: float
    low_beta_starts: int
    high_beta_starts: int

    @property
    def h(self) -> float:
        return 1.0 / self.n_steps

    @property
    def abs_z_grid(self) -> np.ndarray:
        return np.unique(np.round(np.abs(self.z_grid), 12))

    def payload(self) -> dict[str, Any]:
        """Every numerical choice which makes a checkpoint compatible."""
        return {
            "schema_version": SCHEMA_VERSION,
            "n_steps": self.n_steps,
            "z_grid": self.z_grid.tolist(),
            "v_table_cap": self.v_table_cap,
            "v_bound": self.v_bound,
            "maxiter": self.maxiter,
            "retry_maxiter": self.retry_maxiter,
            "ftol": self.ftol,
            "low_beta_starts": self.low_beta_starts,
            "high_beta_starts": self.high_beta_starts,
            "constraint_tol": CONSTRAINT_TOL,
            "objective_tol": OBJECTIVE_TOL,
            "rate_table": {
                "edge_points": 1200,
                "core_points": 2401,
                "stable_bessel": "i0e/i1e",
            },
        }

    @property
    def config_json(self) -> str:
        return json.dumps(self.payload(), sort_keys=True, separators=(",", ":"))

    @property
    def config_hash(self) -> str:
        return hashlib.sha256(self.config_json.encode("utf-8")).hexdigest()


@dataclass
class CandidateResult:
    v: np.ndarray
    value: float
    residual: float
    success: bool
    nit: int
    message: str
    source: str
    retried: bool


@dataclass
class FrameResult:
    f_frame: np.ndarray
    profiles_abs: np.ndarray
    success_abs: np.ndarray
    residual_abs: np.ndarray
    nit_abs: np.ndarray
    objective_abs: np.ndarray
    candidate_count_abs: np.ndarray
    retry_abs: np.ndarray
    message_abs: np.ndarray
    elapsed_seconds: float


@dataclass
class Checkpoint:
    index: int
    wi: float
    beta: float
    frame: FrameResult


class FrameSolveError(RuntimeError):
    """A frame did not yield a trustworthy globally selected candidate."""


class StableRateFunction:
    """Legendre rate function for one unit-circle increment.

    ``i0e`` and ``i1e`` preserve precision for large lambda without the
    discontinuous leading-asymptotic switch used by the historical script.
    """

    def __init__(self, table_cap: float) -> None:
        self.table_cap = float(table_cap)
        self.v_table, self.l_table = self._build_table()

    @staticmethod
    def a_scalar(lam: float) -> float:
        x = abs(float(lam))
        if x < 1.0e-7:
            return x**2 / 4.0 - x**4 / 64.0 + x**6 / 576.0
        return float(x + np.log(i0e(x)))

    @staticmethod
    def aprime_scalar(lam: float) -> float:
        lam = float(lam)
        x = abs(lam)
        if x < 1.0e-7:
            return np.copysign(x / 2.0 - x**3 / 16.0 + x**5 / 96.0, lam)
        ratio = float(i1e(x) / i0e(x))
        return ratio if lam >= 0.0 else -ratio

    def l_of_v(self, v: float) -> float:
        av = abs(float(v))
        if av < 1.0e-12:
            return 0.0
        if av >= 1.0:
            return np.inf

        left, right = 1.0e-12, 1.0
        while self.aprime_scalar(right) <= av:
            right *= 2.0
            if right > 2.0e7:
                raise RuntimeError(f"Could not bracket lambda for |v|={av:.12g}.")

        root = root_scalar(
            lambda lam: self.aprime_scalar(lam) - av,
            bracket=(left, right),
            method="brentq",
        )
        if not root.converged:
            raise RuntimeError(f"Legendre inversion failed for |v|={av:.12g}.")
        lam = float(root.root)
        return lam * av - self.a_scalar(lam)

    def _build_table(self) -> tuple[np.ndarray, np.ndarray]:
        # The dense geometric edge resolves the logarithmic divergence as
        # |v| approaches one.  Its end point is finite and intentionally lies
        # outside the optimizer's bounds (see v_bound in SolverConfig).
        edge_distance = np.geomspace(1.0e-6, 2.0e-2, 1200)
        v_edge = 1.0 - edge_distance
        if not np.isclose(v_edge[0], self.table_cap):
            raise RuntimeError("Rate-table edge no longer agrees with table cap.")
        v_core = np.linspace(-0.98, 0.98, 2401)
        v_table = np.unique(np.concatenate((-v_edge[::-1], v_core, v_edge)))
        l_table = np.array([self.l_of_v(v) for v in v_table], dtype=float)
        if not np.all(np.isfinite(l_table)):
            raise RuntimeError("The rate-function table contains non-finite values.")
        return v_table, l_table

    def interpolate(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        abs_v = np.abs(v)
        values = np.interp(v, self.v_table, self.l_table)
        # The strict test retains a finite objective at the tabulated endpoint.
        # The optimizer itself uses a smaller bound, so this is only a robust
        # guard against an invalid external caller.
        return np.where(abs_v > self.table_cap + 1.0e-12, np.inf, values)


class VariationalSolver:
    """Global-by-multistart constrained solver for one beta frame."""

    def __init__(self, config: SolverConfig) -> None:
        self.config = config
        self.rate = StableRateFunction(config.v_table_cap)
        self.n = config.n_steps
        self.h = config.h
        self.midpoints = (np.arange(self.n) + 0.5) / self.n
        self.bounds = [(-config.v_bound, config.v_bound)] * self.n

        rounded_abs = np.round(np.abs(config.z_grid), 12)
        self.abs_indices = np.searchsorted(config.abs_z_grid, rounded_abs)
        if not np.allclose(config.abs_z_grid[self.abs_indices], rounded_abs, atol=0.0, rtol=0.0):
            raise RuntimeError("Could not map the z grid to its absolute-value grid.")

    def energy_from_v(self, v: np.ndarray) -> float:
        m = np.zeros(self.n + 1, dtype=float)
        m[1:] = self.h * np.cumsum(v)
        mbar = self.h * (0.5 * m[0] + np.sum(m[1:-1]) + 0.5 * m[-1])
        return float(
            self.h
            * (
                0.5 * (m[0] - mbar) ** 2
                + np.sum((m[1:-1] - mbar) ** 2)
                + 0.5 * (m[-1] - mbar) ** 2
            )
        )

    def objective(self, v: np.ndarray, beta: float) -> float:
        entropy = self.rate.interpolate(v)
        if not np.all(np.isfinite(entropy)):
            return -np.inf
        return float(beta * self.energy_from_v(v) - self.h * np.sum(entropy))

    def project_to_mean(self, v: np.ndarray, z_abs: float, max_iter: int = 40) -> np.ndarray:
        """Water-fill an initial profile onto h*sum(v)=z_abs inside bounds."""
        v = np.clip(np.asarray(v, dtype=float).copy(), -self.config.v_bound, self.config.v_bound)
        target_sum = self.n * z_abs
        for _ in range(max_iter):
            diff = target_sum - float(np.sum(v))
            if abs(diff) < 1.0e-11:
                break
            free = (v > -self.config.v_bound + 1.0e-10) & (
                v < self.config.v_bound - 1.0e-10
            )
            if not np.any(free):
                break
            v[free] += diff / int(np.count_nonzero(free))
            v = np.clip(v, -self.config.v_bound, self.config.v_bound)
        residual = abs(self.h * float(np.sum(v)) - z_abs)
        if residual > CONSTRAINT_TOL / 10.0:
            raise FrameSolveError(
                f"Could not project initial profile at |z|={z_abs:.6g}; residual={residual:.3e}."
            )
        return v

    def _mode(self, k: int) -> np.ndarray:
        mode = np.cos(2.0 * np.pi * k * self.midpoints)
        return mode / np.max(np.abs(mode))

    def _unique_starts(self, starts: list[tuple[str, np.ndarray]]) -> list[tuple[str, np.ndarray]]:
        unique: list[tuple[str, np.ndarray]] = []
        for source, start in starts:
            if not any(np.linalg.norm(start - old, ord=np.inf) < 1.0e-9 for _, old in unique):
                unique.append((source, start))
        return unique

    def make_starts(
        self,
        z_abs: float,
        beta: float,
        prev_z_profile: np.ndarray | None,
        prior_beta_profile: np.ndarray | None,
    ) -> list[tuple[str, np.ndarray]]:
        """Build deterministic starts, then retain the best screened ones."""
        base = self.project_to_mean(np.full(self.n, z_abs), z_abs)
        starts: list[tuple[str, np.ndarray]] = [("uniform", base)]

        if prev_z_profile is not None:
            starts.append(("previous_abs_z", self.project_to_mean(prev_z_profile, z_abs)))
        if prior_beta_profile is not None:
            starts.append(("previous_beta", self.project_to_mean(prior_beta_profile, z_abs)))

        # A small first harmonic gets off the stationary uniform branch just
        # above 4*pi^2.  At high beta, several harmonics/amplitudes also guard
        # against solver basin dependence.
        modes = (1,) if beta < BETA_NONUNIFORM_ONSET else (1, 2, 3)
        amplitudes = (0.10,) if beta < BETA_NONUNIFORM_ONSET else (0.025, 0.10, 0.40, 0.80)
        for k in modes:
            mode = self._mode(k)
            for amplitude in amplitudes:
                starts.append(
                    (
                        f"cos{k}_plus_{amplitude:g}",
                        self.project_to_mean(base + amplitude * mode, z_abs),
                    )
                )
                starts.append(
                    (
                        f"cos{k}_minus_{amplitude:g}",
                        self.project_to_mean(base - amplitude * mode, z_abs),
                    )
                )

        unique = self._unique_starts(starts)
        scored = [
            (self.objective(start, beta), source_order, source, start)
            for source_order, (source, start) in enumerate(unique)
        ]
        # Sorting source_order second guarantees deterministic tie breaking.
        scored.sort(key=lambda item: (-item[0], item[1]))
        count = (
            self.config.high_beta_starts
            if beta >= BETA_NONUNIFORM_ONSET
            else self.config.low_beta_starts
        )
        count = min(max(1, count), len(scored))
        return [(source, start) for _, _, source, start in scored[:count]]

    def _minimize_once(
        self,
        start: np.ndarray,
        z_abs: float,
        beta: float,
        maxiter: int,
    ) -> CandidateResult:
        constraints = [{"type": "eq", "fun": lambda v: self.h * np.sum(v) - z_abs}]

        def neg_objective(v: np.ndarray) -> float:
            value = self.objective(v, beta)
            # The bounds are intentionally inside the tabulated rate domain;
            # retain a finite penalty nevertheless so finite differences never
            # produce NaNs if a backend probes an invalid point.
            return -value if np.isfinite(value) else 1.0e100

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Values in x were outside bounds during a minimize step",
                category=RuntimeWarning,
            )
            result = minimize(
                neg_objective,
                start,
                method="SLSQP",
                bounds=self.bounds,
                constraints=constraints,
                options={"maxiter": maxiter, "ftol": self.config.ftol, "disp": False},
            )

        v = np.asarray(result.x, dtype=float)
        residual = abs(self.h * float(np.sum(v)) - z_abs)
        value = self.objective(v, beta)
        valid = (
            np.isfinite(value)
            and residual <= CONSTRAINT_TOL
            and np.all(np.abs(v) <= self.config.v_bound + 1.0e-10)
        )
        return CandidateResult(
            v=v,
            value=float(value),
            residual=float(residual),
            success=bool(result.success and valid),
            nit=int(getattr(result, "nit", -1)),
            message=str(result.message),
            source="",
            retried=False,
        )

    def optimize_start(
        self,
        source: str,
        start: np.ndarray,
        z_abs: float,
        beta: float,
    ) -> CandidateResult:
        candidate = self._minimize_once(start, z_abs, beta, self.config.maxiter)
        candidate.source = source
        if candidate.success:
            return candidate

        # A finite, feasible non-success status is not treated as a result:
        # retry from it with a larger budget and retain the explicit status.
        retry_start = self.project_to_mean(candidate.v, z_abs)
        retry = self._minimize_once(retry_start, z_abs, beta, self.config.retry_maxiter)
        retry.source = source
        retry.retried = True
        retry.nit += max(candidate.nit, 0)
        retry.message = f"first: {candidate.message}; retry: {retry.message}"
        return retry

    def solve_abs_z(
        self,
        z_abs: float,
        beta: float,
        prev_z_profile: np.ndarray | None,
        prior_beta_profile: np.ndarray | None,
    ) -> tuple[CandidateResult, int]:
        starts = self.make_starts(z_abs, beta, prev_z_profile, prior_beta_profile)
        candidates = [self.optimize_start(source, start, z_abs, beta) for source, start in starts]
        converged = [candidate for candidate in candidates if candidate.success]
        if not converged:
            diagnostic = "; ".join(
                f"{c.source}: {c.message} (res={c.residual:.2e}, value={c.value:.6g})"
                for c in candidates
            )
            raise FrameSolveError(
                f"No converged candidate at beta={beta:.9g}, |z|={z_abs:.6g}: {diagnostic}"
            )

        best = max(converged, key=lambda candidate: candidate.value)
        nonconverged = [
            candidate
            for candidate in candidates
            if not candidate.success and np.isfinite(candidate.value)
        ]
        if nonconverged:
            best_unconverged = max(nonconverged, key=lambda candidate: candidate.value)
            if best_unconverged.value > best.value + OBJECTIVE_TOL:
                raise FrameSolveError(
                    "A non-converged candidate outranks every converged candidate at "
                    f"beta={beta:.9g}, |z|={z_abs:.6g}: "
                    f"{best_unconverged.source}={best_unconverged.value:.9g} vs "
                    f"{best.source}={best.value:.9g}."
                )
        return best, len(starts)

    def solve_frame(self, beta: float, prior_profiles: np.ndarray | None = None) -> FrameResult:
        t0 = perf_counter()
        abs_z_grid = self.config.abs_z_grid
        n_abs = len(abs_z_grid)
        profiles = np.empty((n_abs, self.n), dtype=float)
        objectives = np.empty(n_abs, dtype=float)
        success = np.zeros(n_abs, dtype=bool)
        residuals = np.full(n_abs, np.nan, dtype=float)
        nits = np.full(n_abs, -1, dtype=int)
        candidate_counts = np.zeros(n_abs, dtype=int)
        retried = np.zeros(n_abs, dtype=bool)
        messages = np.full(n_abs, "", dtype="<U320")

        if prior_profiles is not None and prior_profiles.shape != profiles.shape:
            raise ValueError(
                f"Prior profile shape {prior_profiles.shape} does not match {profiles.shape}."
            )

        prev_z_profile: np.ndarray | None = None
        for j, z_abs in enumerate(abs_z_grid):
            prior = None if prior_profiles is None else prior_profiles[j]
            best, count = self.solve_abs_z(
                float(z_abs), beta, prev_z_profile=prev_z_profile, prior_beta_profile=prior
            )
            profiles[j] = best.v
            objectives[j] = best.value
            success[j] = best.success
            residuals[j] = best.residual
            nits[j] = best.nit
            candidate_counts[j] = count
            retried[j] = best.retried
            messages[j] = f"{best.source}: {best.message}"[:320]
            prev_z_profile = best.v.copy()

        f_frame = objectives[self.abs_indices]
        if not np.all(np.isfinite(f_frame)) or not np.all(success):
            raise FrameSolveError(f"Frame beta={beta:.9g} contains an invalid solution.")
        return FrameResult(
            f_frame=f_frame,
            profiles_abs=profiles,
            success_abs=success,
            residual_abs=residuals,
            nit_abs=nits,
            objective_abs=objectives,
            candidate_count_abs=candidate_counts,
            retry_abs=retried,
            message_abs=messages,
            elapsed_seconds=perf_counter() - t0,
        )


def checkpoint_path(checkpoint_dir: Path, index: int, wi: float) -> Path:
    return checkpoint_dir / f"frame_{index:03d}_wi_{wi:.6f}.npz"


def atomic_savez(path: Path, **arrays: Any) -> None:
    """Write an NPZ completely before atomically making it visible."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        # Flush the directory entry where supported (Linux/NFS-safe best effort).
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _scalar_text(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


def load_valid_checkpoint(
    path: Path,
    config: SolverConfig,
    index: int,
    wi: float,
    beta: float,
) -> Checkpoint | None:
    """Return a checkpoint only if it is fully compatible and trustworthy."""
    if not path.exists():
        return None
    abs_z = config.abs_z_grid
    required = {
        "schema_version",
        "config_hash",
        "index",
        "wi",
        "beta",
        "z_grid",
        "abs_z_grid",
        "f_frame",
        "profiles_abs",
        "success_abs",
        "residual_abs",
        "nit_abs",
        "objective_abs",
        "candidate_count_abs",
        "retry_abs",
        "message_abs",
        "complete",
        "elapsed_seconds",
    }
    try:
        with np.load(path, allow_pickle=False) as data:
            if not required.issubset(data.files):
                return None
            if _scalar_text(data["schema_version"]) != SCHEMA_VERSION:
                return None
            if _scalar_text(data["config_hash"]) != config.config_hash:
                return None
            if int(np.asarray(data["index"]).item()) != index:
                return None
            if not bool(np.asarray(data["complete"]).item()):
                return None
            if not np.isclose(float(np.asarray(data["wi"]).item()), wi, rtol=0.0, atol=1.0e-12):
                return None
            if not np.isclose(float(np.asarray(data["beta"]).item()), beta, rtol=0.0, atol=1.0e-10):
                return None
            z_grid = np.asarray(data["z_grid"], dtype=float)
            abs_z_grid = np.asarray(data["abs_z_grid"], dtype=float)
            f_frame = np.asarray(data["f_frame"], dtype=float)
            profiles_abs = np.asarray(data["profiles_abs"], dtype=float)
            success_abs = np.asarray(data["success_abs"], dtype=bool)
            residual_abs = np.asarray(data["residual_abs"], dtype=float)
            nit_abs = np.asarray(data["nit_abs"], dtype=int)
            objective_abs = np.asarray(data["objective_abs"], dtype=float)
            candidate_count_abs = np.asarray(data["candidate_count_abs"], dtype=int)
            retry_abs = np.asarray(data["retry_abs"], dtype=bool)
            message_abs = np.asarray(data["message_abs"], dtype="<U320")
            elapsed_seconds = float(np.asarray(data["elapsed_seconds"]).item())
    except (OSError, ValueError, KeyError, EOFError):
        return None

    expected_shape = (len(abs_z), config.n_steps)
    if (
        not np.array_equal(z_grid, config.z_grid)
        or not np.array_equal(abs_z_grid, abs_z)
        or f_frame.shape != config.z_grid.shape
        or profiles_abs.shape != expected_shape
        or success_abs.shape != abs_z.shape
        or residual_abs.shape != abs_z.shape
        or nit_abs.shape != abs_z.shape
        or objective_abs.shape != abs_z.shape
        or candidate_count_abs.shape != abs_z.shape
        or retry_abs.shape != abs_z.shape
        or message_abs.shape != abs_z.shape
    ):
        return None
    if (
        not np.all(np.isfinite(f_frame))
        or not np.all(np.isfinite(profiles_abs))
        or not np.all(np.isfinite(residual_abs))
        or not np.all(np.isfinite(objective_abs))
        or not np.all(success_abs)
        or np.any(residual_abs > CONSTRAINT_TOL)
        or np.any(np.abs(profiles_abs) > config.v_bound + 1.0e-10)
        or not np.isfinite(elapsed_seconds)
    ):
        return None

    frame = FrameResult(
        f_frame=f_frame,
        profiles_abs=profiles_abs,
        success_abs=success_abs,
        residual_abs=residual_abs,
        nit_abs=nit_abs,
        objective_abs=objective_abs,
        candidate_count_abs=candidate_count_abs,
        retry_abs=retry_abs,
        message_abs=message_abs,
        elapsed_seconds=elapsed_seconds,
    )
    return Checkpoint(index=index, wi=wi, beta=beta, frame=frame)


def save_checkpoint(
    path: Path,
    config: SolverConfig,
    index: int,
    wi: float,
    beta: float,
    frame: FrameResult,
) -> None:
    atomic_savez(
        path,
        schema_version=np.asarray(SCHEMA_VERSION),
        config_hash=np.asarray(config.config_hash),
        config_json=np.asarray(config.config_json),
        index=np.asarray(index, dtype=np.int64),
        wi=np.asarray(wi, dtype=float),
        beta=np.asarray(beta, dtype=float),
        z_grid=config.z_grid,
        abs_z_grid=config.abs_z_grid,
        f_frame=frame.f_frame,
        profiles_abs=frame.profiles_abs,
        success_abs=frame.success_abs,
        residual_abs=frame.residual_abs,
        nit_abs=frame.nit_abs,
        objective_abs=frame.objective_abs,
        candidate_count_abs=frame.candidate_count_abs,
        retry_abs=frame.retry_abs,
        message_abs=frame.message_abs,
        elapsed_seconds=np.asarray(frame.elapsed_seconds, dtype=float),
        complete=np.asarray(True),
        saved_utc=np.asarray(datetime.now(UTC).isoformat()),
    )


def save_failure(path: Path, index: int, wi: float, beta: float, error: BaseException) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "index": index,
        "wi": wi,
        "beta": beta,
        "saved_utc": datetime.now(UTC).isoformat(),
        "error": str(error),
        "traceback": traceback.format_exc(),
    }
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_aggregate(
    output_root: Path,
    config: SolverConfig,
    wi_values: np.ndarray,
    completed: dict[int, Checkpoint],
) -> Path:
    """Publish an atomically replaced partial or complete aggregate."""
    n_frames = len(wi_values)
    n_abs = len(config.abs_z_grid)
    f_all = np.full((n_frames, len(config.z_grid)), np.nan, dtype=float)
    complete_mask = np.zeros(n_frames, dtype=bool)
    success = np.zeros((n_frames, n_abs), dtype=bool)
    residuals = np.full((n_frames, n_abs), np.nan, dtype=float)
    nits = np.full((n_frames, n_abs), -1, dtype=int)
    objectives = np.full((n_frames, n_abs), np.nan, dtype=float)
    candidate_counts = np.zeros((n_frames, n_abs), dtype=int)
    retries = np.zeros((n_frames, n_abs), dtype=bool)
    elapsed = np.full(n_frames, np.nan, dtype=float)
    for index, checkpoint in completed.items():
        frame = checkpoint.frame
        f_all[index] = frame.f_frame
        complete_mask[index] = True
        success[index] = frame.success_abs
        residuals[index] = frame.residual_abs
        nits[index] = frame.nit_abs
        objectives[index] = frame.objective_abs
        candidate_counts[index] = frame.candidate_count_abs
        retries[index] = frame.retry_abs
        elapsed[index] = frame.elapsed_seconds

    # This stable public name is consumed by the combined Wi plotter.  The
    # configuration fingerprint inside prevents a changed N/z grid from being
    # mistaken for compatible data.
    aggregate_path = output_root / "ninf_wi_001_10.npz"
    atomic_savez(
        aggregate_path,
        schema_version=np.asarray(SCHEMA_VERSION),
        config_hash=np.asarray(config.config_hash),
        config_json=np.asarray(config.config_json),
        wi_values=wi_values,
        beta_values=wi_values * BETA_PER_WI,
        z_grid=config.z_grid,
        abs_z_grid=config.abs_z_grid,
        F_all=f_all,
        completed=complete_mask,
        # Lower-case aliases retain a convenient direct representation for
        # command-line inspection while the two fields above are the public
        # plotting contract.
        f_all=f_all,
        complete=complete_mask,
        all_complete=np.asarray(bool(np.all(complete_mask))),
        success_abs=success,
        residual_abs=residuals,
        nit_abs=nits,
        objective_abs=objectives,
        candidate_count_abs=candidate_counts,
        retry_abs=retries,
        elapsed_seconds=elapsed,
        updated_utc=np.asarray(datetime.now(UTC).isoformat()),
    )
    return aggregate_path


def write_progress(output_root: Path, wi_values: np.ndarray, completed: dict[int, Checkpoint]) -> None:
    lines = ["index\tWi\tbeta\tstatus\telapsed_seconds\tmax_residual"]
    for index, wi in enumerate(wi_values):
        beta = wi * BETA_PER_WI
        checkpoint = completed.get(index)
        if checkpoint is None:
            lines.append(f"{index}\t{wi:.6f}\t{beta:.12g}\tpending\t\t")
            continue
        frame = checkpoint.frame
        lines.append(
            f"{index}\t{wi:.6f}\t{beta:.12g}\tcomplete\t"
            f"{frame.elapsed_seconds:.6f}\t{np.max(frame.residual_abs):.3e}"
        )
    atomic_write_text(output_root / "progress.tsv", "\n".join(lines) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--n-steps",
        type=int,
        default=int(os.environ.get("N_INF_STEPS", "80")),
        help="Path discretization N (default: N_INF_STEPS or 80).",
    )
    parser.add_argument("--z-points", type=int, default=39)
    parser.add_argument("--z-max", type=float, default=0.95)
    parser.add_argument("--maxiter", type=int, default=800)
    parser.add_argument("--retry-maxiter", type=int, default=1600)
    parser.add_argument("--ftol", type=float, default=1.0e-10)
    parser.add_argument(
        "--low-beta-starts",
        type=int,
        default=1,
        help="Number of screened deterministic starts below 4*pi^2 (default: 1).",
    )
    parser.add_argument("--high-beta-starts", type=int, default=6)
    parser.add_argument(
        "--index",
        type=int,
        action="append",
        help="Compute only this fixed-grid index (repeatable; useful for a probe).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="At most this many pending selected frames; useful for a lightweight check.",
    )
    parser.add_argument("--force", action="store_true", help="Recompute selected frames even if valid.")
    parser.add_argument("--no-aggregate", action="store_true", help="Do not refresh aggregate/progress files.")
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop at the first failed frame instead of continuing with later points.",
    )
    args = parser.parse_args(argv)
    if args.n_steps < 4:
        parser.error("--n-steps must be at least 4.")
    if args.maxiter < 1 or args.retry_maxiter < args.maxiter:
        parser.error("Require 1 <= --maxiter <= --retry-maxiter.")
    if args.low_beta_starts < 1 or args.high_beta_starts < 1:
        parser.error("The multistart counts must be positive.")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive.")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    wi_values = make_wi_grid()
    z_grid = make_z_grid(args.z_points, args.z_max)
    # Leave an explicit interior buffer: the rate table includes 1-1e-6 while
    # SLSQP is bounded at 1-2e-6, preventing its finite differences from
    # sampling a point assigned an infinite entropy.
    config = SolverConfig(
        n_steps=args.n_steps,
        z_grid=z_grid,
        v_table_cap=1.0 - 1.0e-6,
        v_bound=1.0 - 2.0e-6,
        maxiter=args.maxiter,
        retry_maxiter=args.retry_maxiter,
        ftol=args.ftol,
        low_beta_starts=args.low_beta_starts,
        high_beta_starts=args.high_beta_starts,
    )
    output_root = args.output_root.resolve()
    checkpoint_dir = output_root / "checkpoints"
    failures_dir = output_root / "failures"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    failures_dir.mkdir(parents=True, exist_ok=True)

    all_indices = list(range(len(wi_values)))
    if args.index:
        requested = sorted(set(args.index))
        invalid = [index for index in requested if index < 0 or index >= len(wi_values)]
        if invalid:
            raise SystemExit(f"Grid indices out of range 0..{len(wi_values) - 1}: {invalid}")
        selected_indices = requested
    else:
        selected_indices = all_indices

    completed: dict[int, Checkpoint] = {}
    for index, wi in enumerate(wi_values):
        beta = float(wi * BETA_PER_WI)
        existing = load_valid_checkpoint(
            checkpoint_path(checkpoint_dir, index, float(wi)), config, index, float(wi), beta
        )
        if existing is not None:
            completed[index] = existing

    pending = [index for index in selected_indices if args.force or index not in completed]
    if args.max_frames is not None:
        pending = pending[: args.max_frames]

    print(
        f"n->infinity Wi sweep: {len(wi_values)} points, Wi=[{wi_values[0]:.6g},{wi_values[-1]:.6g}], "
        f"beta=[{wi_values[0] * BETA_PER_WI:.6g},{wi_values[-1] * BETA_PER_WI:.6g}]",
        flush=True,
    )
    print(
        f"N={config.n_steps}, z={len(config.z_grid)} points, "
        f"high-beta onset Wi={BETA_NONUNIFORM_ONSET * WI_PER_BETA:.9g}; "
        f"valid checkpoints={len(completed)}, pending={len(pending)}",
        flush=True,
    )

    if not pending:
        if not args.no_aggregate:
            aggregate = write_aggregate(output_root, config, wi_values, completed)
            write_progress(output_root, wi_values, completed)
            print(f"No frames need work; aggregate refreshed -> {aggregate}", flush=True)
        return 0

    print("Building stable i0e/i1e Legendre rate table...", flush=True)
    solver = VariationalSolver(config)
    print("Rate table ready.", flush=True)

    failures: list[int] = []
    for ordinal, index in enumerate(pending, start=1):
        wi = float(wi_values[index])
        beta = wi * BETA_PER_WI
        prior_profiles = None
        for previous in range(index - 1, -1, -1):
            prior = completed.get(previous)
            if prior is not None:
                prior_profiles = prior.frame.profiles_abs
                break

        print(
            f"[{ordinal}/{len(pending)}] index={index:03d} Wi={wi:.6f} beta={beta:.12g}",
            flush=True,
        )
        try:
            frame = solver.solve_frame(beta, prior_profiles=prior_profiles)
            target = checkpoint_path(checkpoint_dir, index, wi)
            save_checkpoint(target, config, index, wi, beta, frame)
            completed[index] = Checkpoint(index=index, wi=wi, beta=beta, frame=frame)
            print(
                f"  complete in {frame.elapsed_seconds:.2f}s; "
                f"max residual={np.max(frame.residual_abs):.2e}; -> {target}",
                flush=True,
            )
            if not args.no_aggregate:
                aggregate = write_aggregate(output_root, config, wi_values, completed)
                write_progress(output_root, wi_values, completed)
                print(f"  aggregate ({len(completed)}/{len(wi_values)}) -> {aggregate}", flush=True)
        except Exception as error:
            failures.append(index)
            failure_path = failures_dir / f"frame_{index:03d}_wi_{wi:.6f}.json"
            save_failure(failure_path, index, wi, beta, error)
            print(f"  FAILED: {error}\n  diagnostics -> {failure_path}", file=sys.stderr, flush=True)
            if args.stop_on_error:
                break

    if not args.no_aggregate:
        aggregate = write_aggregate(output_root, config, wi_values, completed)
        write_progress(output_root, wi_values, completed)
        print(f"Final aggregate refresh -> {aggregate}", flush=True)

    if failures:
        print(f"{len(failures)} frame(s) failed and were left uncheckpointed: {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
