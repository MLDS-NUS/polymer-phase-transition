#!/usr/bin/env python3
"""Resumable :math:`n -> infinity` joint ``z2=0, z3`` sweep on Wi=0.02..10.

This is a new, self-contained runner; it neither imports nor writes the older
``Inf_limt`` beta sweeps.  A completed Wi frame contains the centred optimizer
and six independently optimized positive-z3 slope profiles for the tilted-MC
proposal mixture.  Frames are written atomically and can be processed in any
order.  The aggregate is deliberately partial-safe: unfinished rows are NaN
and indicated by ``completed``.

The plotted continuum quantity is the *maximizer set*, not a noisy raw argmax.
At z2=0, the shift theorem makes its edges

    [-2 ||w*||_infinity, +2 ||w*||_infinity]

when beta > 4 pi^2, where w* is the centred z3=0 optimum.  We publish exactly
zero for both edges at and below the onset rather than exposing round-off sized
nonzero amplitudes there.

Examples
--------
Print the fully fingerprinted numerical configuration without creating data::

    python run_ninf_joint_z3_wi_001_10.py --config

Run representative endpoint/onset/tail checks::

    python -u run_ninf_joint_z3_wi_001_10.py --preflight

Test a single default-resolution checkpoint::

    python -u run_ninf_joint_z3_wi_001_10.py --index 0 --max-frames 1
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
import traceback
import uuid
import warnings
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from scipy.optimize import LinearConstraint, minimize, root_scalar
from scipy.special import i0e, i1e

from common import (
    ANCHOR_Z3,
    BETA_PER_WI,
    WI_PER_BETA,
    atomic_savez,
    atomic_write_text,
    beta_from_wi,
    canonical_json,
    config_hash,
    make_wi_grid,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = HERE / "data" / "ninf"

SCHEMA_VERSION = "ninf-joint-z3-wi-001-10-v1"
CONFIG_VERSION = 1
Z2_FIXED = 0.0
BETA_PLATEAU_ONSET = 4.0 * np.pi**2
CONSTRAINT_TOL = 2.0e-7
OBJECTIVE_TOL = 1.0e-7


class FrameSolveError(RuntimeError):
    """Raised when a frame has no certified constrained optimum."""


@dataclass(frozen=True)
class SolverConfig:
    """All numerical choices which make a checkpoint compatible."""

    n_steps: int
    anchor_z3: np.ndarray
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

    def payload(self) -> dict[str, Any]:
        wi = make_wi_grid()
        return {
            "schema_version": SCHEMA_VERSION,
            "config_version": CONFIG_VERSION,
            "model": "mogulskii_joint_z2_0_unrestricted",
            "z2_fixed": Z2_FIXED,
            "wi_values": wi.tolist(),
            "wi_per_beta": WI_PER_BETA,
            "beta_per_wi": BETA_PER_WI,
            "beta_plateau_onset": BETA_PLATEAU_ONSET,
            "maximizer_set_rule": "zero_at_or_below_onset; +/-2*||w_star||_inf_above",
            "n_steps": self.n_steps,
            "anchor_z3": self.anchor_z3.tolist(),
            "v_table_cap": self.v_table_cap,
            "v_bound": self.v_bound,
            "maxiter": self.maxiter,
            "retry_maxiter": self.retry_maxiter,
            "ftol": self.ftol,
            "low_beta_starts": self.low_beta_starts,
            "high_beta_starts": self.high_beta_starts,
            "constraint_tol": CONSTRAINT_TOL,
            "objective_tolerance": OBJECTIVE_TOL,
            "rate_table": {
                "edge_points": 1200,
                "core_points": 2401,
                "bessel": "scipy.special.i0e/i1e (scaled)",
            },
        }

    @property
    def config_json(self) -> str:
        return canonical_json(self.payload())

    @property
    def fingerprint(self) -> str:
        return config_hash(self.payload())


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
    anchor_profiles: np.ndarray
    anchor_objective: np.ndarray
    anchor_residual: np.ndarray
    anchor_success: np.ndarray
    anchor_nit: np.ndarray
    anchor_candidate_count: np.ndarray
    anchor_retry: np.ndarray
    anchor_message: np.ndarray
    center_profile: np.ndarray
    center_objective: float
    center_residual: float
    center_success: bool
    raw_center_extent: float
    center_extent: float
    edge_lo: float
    edge_hi: float
    above_onset: bool
    elapsed_seconds: float


@dataclass
class Checkpoint:
    index: int
    wi: float
    beta: float
    frame: FrameResult


class StableBesselRate:
    """Stable Legendre rate function for a unit-circle increment.

    ``log(I0(x))`` is evaluated as ``x + log(i0e(x))`` and ``I1/I0`` as
    ``i1e/i0e``.  This remains finite well beyond the high-Wi range, unlike
    the direct historical ``i0``/``i1`` implementation.
    """

    EDGE_POINTS = 1200
    CORE_POINTS = 2401

    def __init__(self, table_cap: float) -> None:
        self.table_cap = float(table_cap)
        self.v_table, self.l_table = self._build_table()

    @staticmethod
    def log_i0_scalar(lam: float) -> float:
        x = abs(float(lam))
        if x < 1.0e-7:
            return x * x / 4.0 - x**4 / 64.0 + x**6 / 576.0
        return float(x + np.log(i0e(x)))

    @staticmethod
    def mean_resultant_scalar(lam: float) -> float:
        lam = float(lam)
        x = abs(lam)
        if x < 1.0e-7:
            return float(np.copysign(x / 2.0 - x**3 / 16.0 + x**5 / 96.0, lam))
        ratio = float(i1e(x) / i0e(x))
        return ratio if lam >= 0.0 else -ratio

    def lambda_for_mean(self, v: float) -> float:
        """Invert I1(lambda)/I0(lambda)=v with a bracketing solve."""
        value = float(v)
        magnitude = abs(value)
        if magnitude < 1.0e-12:
            return 0.0
        if magnitude >= 1.0:
            raise ValueError(f"A finite von-Mises tilt cannot have mean {value!r}.")
        upper = 1.0
        while self.mean_resultant_scalar(upper) <= magnitude:
            upper *= 2.0
            if upper > 2.0e7:
                raise RuntimeError(f"Could not bracket Bessel inverse for |v|={magnitude:.12g}.")
        result = root_scalar(
            lambda lam: self.mean_resultant_scalar(lam) - magnitude,
            bracket=(1.0e-12, upper),
            method="brentq",
            xtol=1.0e-12,
        )
        if not result.converged:
            raise RuntimeError(f"Bessel inverse failed for |v|={magnitude:.12g}.")
        return float(np.copysign(result.root, value))

    def rate_scalar(self, v: float) -> float:
        magnitude = abs(float(v))
        if magnitude < 1.0e-12:
            return 0.0
        if magnitude >= 1.0:
            return np.inf
        lam = self.lambda_for_mean(magnitude)
        return float(lam * magnitude - self.log_i0_scalar(lam))

    def _build_table(self) -> tuple[np.ndarray, np.ndarray]:
        # The geometric edge resolves the logarithmic divergence near |v|=1.
        # It reaches 1-1e-6; the optimizer stays one further 1e-6 inside it.
        edge_distance = np.geomspace(1.0e-6, 2.0e-2, self.EDGE_POINTS)
        v_edge = 1.0 - edge_distance
        if not np.isclose(v_edge[0], self.table_cap, rtol=0.0, atol=2.0e-16):
            raise RuntimeError("Rate table cap and its edge grid disagree.")
        v_core = np.linspace(-0.98, 0.98, self.CORE_POINTS)
        v_table = np.unique(np.concatenate((-v_edge[::-1], v_core, v_edge)))
        l_table = np.array([self.rate_scalar(v) for v in v_table], dtype=float)
        if not np.all(np.isfinite(l_table)):
            raise RuntimeError("The stable Bessel rate table contains non-finite values.")
        return v_table, l_table

    def interpolate(self, v: np.ndarray) -> np.ndarray:
        values = np.asarray(v, dtype=float)
        out = np.interp(values, self.v_table, self.l_table)
        return np.where(np.abs(values) > self.table_cap + 1.0e-12, np.inf, out)


class VariationalSolver:
    """Constrained multistart solver for the six z3 anchor profiles."""

    def __init__(self, config: SolverConfig) -> None:
        self.config = config
        self.n = config.n_steps
        self.h = config.h
        self.midpoints = (np.arange(self.n, dtype=float) + 0.5) / self.n
        self.bounds = [(-config.v_bound, config.v_bound)] * self.n
        endpoint = np.full(self.n, self.h)
        mean = self.h * (1.0 - self.midpoints)
        self.constraint_matrix = np.vstack((endpoint, mean))
        gram = self.constraint_matrix @ self.constraint_matrix.T
        if np.linalg.matrix_rank(gram) != 2:
            raise RuntimeError("The two equality constraints are not independent.")
        # P maps arbitrary slope perturbations to the exact two-constraint
        # nullspace.  Building starts in this nullspace is much more reliable
        # than clipping a profile and hoping SLSQP repairs both constraints.
        self.null_projector = (
            np.eye(self.n)
            - self.constraint_matrix.T @ np.linalg.solve(gram, self.constraint_matrix)
        )
        self.rate = StableBesselRate(config.v_table_cap)

    def target(self, z3: float) -> np.ndarray:
        return np.array((Z2_FIXED, 0.5 * (Z2_FIXED - float(z3))), dtype=float)

    def constraint_residual(self, v: np.ndarray, z3: float) -> float:
        return float(np.linalg.norm(self.constraint_matrix @ np.asarray(v) - self.target(z3), ord=np.inf))

    def path_from_v(self, v: np.ndarray) -> np.ndarray:
        path = np.zeros(self.n + 1, dtype=float)
        path[1:] = self.h * np.cumsum(np.asarray(v, dtype=float))
        return path

    def energy_from_v(self, v: np.ndarray) -> float:
        path = self.path_from_v(v)
        mean = self.h * (0.5 * path[0] + np.sum(path[1:-1]) + 0.5 * path[-1])
        centered = path - mean
        return float(
            self.h
            * (0.5 * centered[0] ** 2 + np.sum(centered[1:-1] ** 2) + 0.5 * centered[-1] ** 2)
        )

    def objective(self, v: np.ndarray, beta: float) -> float:
        entropy = self.rate.interpolate(v)
        if not np.all(np.isfinite(entropy)):
            return -np.inf
        return float(beta * self.energy_from_v(v) - self.h * np.sum(entropy))

    def triangular_base(self, z3: float) -> np.ndarray:
        """An exactly feasible discrete V-path for the two z2/z3 constraints."""
        z3 = float(z3)
        v = np.where(self.midpoints < 0.5, -2.0 * z3, 2.0 * z3)
        if np.max(np.abs(v)) > self.config.v_bound + 1.0e-12:
            raise FrameSolveError(f"z3={z3:.6g} cannot fit within the slope bounds.")
        residual = self.constraint_residual(v, z3)
        if residual > 5.0e-13:
            raise FrameSolveError(
                f"Discrete triangular start missed the two constraints by {residual:.3e}; "
                "N must be even."
            )
        return v.astype(float, copy=False)

    def _fit_null_perturbation(self, base: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """Add as much of an equality-preserving perturbation as bounds allow."""
        perturbation = self.null_projector @ np.asarray(delta, dtype=float)
        if np.linalg.norm(perturbation, ord=np.inf) < 1.0e-14:
            return base.copy()
        positive = perturbation > 1.0e-15
        negative = perturbation < -1.0e-15
        limits: list[float] = []
        if np.any(positive):
            limits.extend(((self.config.v_bound - base[positive]) / perturbation[positive]).tolist())
        if np.any(negative):
            limits.extend(((-self.config.v_bound - base[negative]) / perturbation[negative]).tolist())
        max_scale = min(limits) if limits else 0.0
        if not np.isfinite(max_scale) or max_scale <= 0.0:
            return base.copy()
        # 0.995 leaves room for finite-difference probes made by SLSQP.
        scale = min(1.0, 0.995 * max_scale)
        candidate = base + scale * perturbation
        if self.constraint_residual(candidate, self._z3_from_target(base)) > 1.0e-10:
            raise FrameSolveError("Nullspace start construction lost an equality constraint.")
        return candidate

    def _z3_from_target(self, v: np.ndarray) -> float:
        # constraint row 1 is (z2-z3)/2 and z2 is fixed at zero.
        return float(-2.0 * (self.constraint_matrix @ v)[1])

    def _mode(self, k: int, phase: str) -> np.ndarray:
        angle = 2.0 * np.pi * k * self.midpoints
        raw = np.cos(angle) if phase == "cos" else np.sin(angle)
        mode = self.null_projector @ raw
        maximum = np.max(np.abs(mode))
        if maximum <= 1.0e-14:
            raise RuntimeError("A proposed multistart mode is not in the nullspace.")
        return mode / maximum

    @staticmethod
    def _unique_starts(starts: list[tuple[str, np.ndarray]]) -> list[tuple[str, np.ndarray]]:
        unique: list[tuple[str, np.ndarray]] = []
        for source, candidate in starts:
            if not any(np.linalg.norm(candidate - prior, ord=np.inf) < 1.0e-10 for _, prior in unique):
                unique.append((source, candidate))
        return unique

    def make_starts(
        self,
        z3: float,
        beta: float,
        previous_anchor: np.ndarray | None,
        prior_beta_anchor: np.ndarray | None,
    ) -> list[tuple[str, np.ndarray]]:
        base = self.triangular_base(z3)
        starts: list[tuple[str, np.ndarray]] = [("triangular", base)]
        for name, profile in (("previous_anchor", previous_anchor), ("previous_beta", prior_beta_anchor)):
            if profile is not None:
                starts.append((name, self._fit_null_perturbation(base, np.asarray(profile) - base)))

        high = beta > BETA_PLATEAU_ONSET
        amplitudes = (0.04, 0.16, 0.48) if not high else (0.02, 0.08, 0.24, 0.55, 0.85)
        modes = (1,) if not high else (1, 2, 3)
        # Cosine is the destabilizing k=1 mode.  Sine phases provide shifted
        # representatives useful for nonzero z3 anchors and prevent a phase
        # convention from locking the solver to an avoidable stationary branch.
        phases = ("cos",) if not high else ("cos", "sin")
        for k in modes:
            for phase in phases:
                mode = self._mode(k, phase)
                for amplitude in amplitudes:
                    starts.append((f"{phase}{k}+{amplitude:g}", self._fit_null_perturbation(base, amplitude * mode)))
                    starts.append((f"{phase}{k}-{amplitude:g}", self._fit_null_perturbation(base, -amplitude * mode)))

        unique = self._unique_starts(starts)
        scored = [
            (self.objective(profile, beta), order, source, profile)
            for order, (source, profile) in enumerate(unique)
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))
        keep = self.config.high_beta_starts if high else self.config.low_beta_starts
        return [(source, profile) for _, _, source, profile in scored[: min(keep, len(scored))]]

    def _minimize_once(self, start: np.ndarray, z3: float, beta: float, maxiter: int) -> CandidateResult:
        target = self.target(z3)
        constraint = LinearConstraint(self.constraint_matrix, target, target)

        def negative_objective(v: np.ndarray) -> float:
            value = self.objective(v, beta)
            return -value if np.isfinite(value) else 1.0e100

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Values in x were outside bounds during a minimize step",
                category=RuntimeWarning,
            )
            result = minimize(
                negative_objective,
                x0=start,
                method="SLSQP",
                bounds=self.bounds,
                constraints=(constraint,),
                options={"maxiter": maxiter, "ftol": self.config.ftol, "disp": False},
            )

        v = np.asarray(result.x, dtype=float)
        residual = self.constraint_residual(v, z3)
        value = self.objective(v, beta)
        valid = (
            np.isfinite(value)
            and np.all(np.isfinite(v))
            and residual <= CONSTRAINT_TOL
            and np.all(np.abs(v) <= self.config.v_bound + 1.0e-10)
        )
        return CandidateResult(
            v=v,
            value=float(value),
            residual=residual,
            success=bool(result.success and valid),
            nit=int(getattr(result, "nit", -1)),
            message=str(result.message),
            source="",
            retried=False,
        )

    def optimize_start(self, source: str, start: np.ndarray, z3: float, beta: float) -> CandidateResult:
        first = self._minimize_once(start, z3, beta, self.config.maxiter)
        first.source = source
        if first.success:
            return first
        repaired = self._fit_null_perturbation(self.triangular_base(z3), first.v - self.triangular_base(z3))
        retry = self._minimize_once(repaired, z3, beta, self.config.retry_maxiter)
        retry.source = source
        retry.retried = True
        retry.nit += max(first.nit, 0)
        retry.message = f"first: {first.message}; retry: {retry.message}"
        return retry

    def solve_anchor(
        self,
        z3: float,
        beta: float,
        previous_anchor: np.ndarray | None,
        prior_beta_anchor: np.ndarray | None,
    ) -> tuple[CandidateResult, int]:
        starts = self.make_starts(z3, beta, previous_anchor, prior_beta_anchor)
        candidates = [self.optimize_start(source, start, z3, beta) for source, start in starts]
        valid = [candidate for candidate in candidates if candidate.success]
        if not valid:
            detail = "; ".join(
                f"{candidate.source}: {candidate.message} (r={candidate.residual:.2e}, J={candidate.value:.8g})"
                for candidate in candidates
            )
            raise FrameSolveError(f"No converged candidate at beta={beta:.10g}, z3={z3:.6g}: {detail}")
        best = max(valid, key=lambda candidate: candidate.value)
        nonconverged = [candidate for candidate in candidates if not candidate.success and np.isfinite(candidate.value)]
        if nonconverged:
            challenger = max(nonconverged, key=lambda candidate: candidate.value)
            if challenger.value > best.value + OBJECTIVE_TOL:
                raise FrameSolveError(
                    f"Unconverged candidate outranks certified optimum at beta={beta:.10g}, z3={z3:.6g}: "
                    f"{challenger.value:.10g} > {best.value:.10g}."
                )
        return best, len(starts)

    def center_extent(self, profile: np.ndarray) -> float:
        path = self.path_from_v(profile)
        mean = self.h * (0.5 * path[0] + np.sum(path[1:-1]) + 0.5 * path[-1])
        w = path - mean
        return float(2.0 * np.max(np.abs(w)))

    def solve_frame(self, beta: float, prior_anchors: np.ndarray | None = None) -> FrameResult:
        if prior_anchors is not None and prior_anchors.shape != (len(self.config.anchor_z3), self.n):
            raise ValueError(
                f"Prior anchor shape {prior_anchors.shape} does not match "
                f"({len(self.config.anchor_z3)}, {self.n})."
            )
        started = perf_counter()
        k = len(self.config.anchor_z3)
        profiles = np.empty((k, self.n), dtype=float)
        objective = np.empty(k, dtype=float)
        residual = np.full(k, np.nan, dtype=float)
        success = np.zeros(k, dtype=bool)
        nit = np.full(k, -1, dtype=int)
        count = np.zeros(k, dtype=int)
        retry = np.zeros(k, dtype=bool)
        message = np.full(k, "", dtype="<U400")

        previous: np.ndarray | None = None
        for anchor_index, z3 in enumerate(self.config.anchor_z3):
            prior = None if prior_anchors is None else prior_anchors[anchor_index]
            best, candidate_count = self.solve_anchor(float(z3), beta, previous, prior)
            profiles[anchor_index] = best.v
            objective[anchor_index] = best.value
            residual[anchor_index] = best.residual
            success[anchor_index] = best.success
            nit[anchor_index] = best.nit
            count[anchor_index] = candidate_count
            retry[anchor_index] = best.retried
            message[anchor_index] = f"{best.source}: {best.message}"[:400]
            previous = best.v.copy()

        if not np.all(success) or not np.all(np.isfinite(profiles)):
            raise FrameSolveError(f"Frame beta={beta:.10g} has invalid anchor results.")
        center = profiles[0].copy()
        raw_extent = self.center_extent(center)
        above = bool(beta > BETA_PLATEAU_ONSET)
        extent = raw_extent if above else 0.0
        return FrameResult(
            anchor_profiles=profiles,
            anchor_objective=objective,
            anchor_residual=residual,
            anchor_success=success,
            anchor_nit=nit,
            anchor_candidate_count=count,
            anchor_retry=retry,
            anchor_message=message,
            center_profile=center,
            center_objective=float(objective[0]),
            center_residual=float(residual[0]),
            center_success=bool(success[0]),
            raw_center_extent=raw_extent,
            center_extent=extent,
            edge_lo=-extent,
            edge_hi=extent,
            above_onset=above,
            elapsed_seconds=perf_counter() - started,
        )


def checkpoint_path(checkpoint_dir: Path, index: int, wi: float) -> Path:
    return checkpoint_dir / f"frame_{index:03d}_wi_{wi:.6f}.npz"


def _scalar_text(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


def _require_scalar_close(data: Any, key: str, expected: float, atol: float) -> bool:
    try:
        return bool(np.isclose(float(np.asarray(data[key]).item()), expected, rtol=0.0, atol=atol))
    except (KeyError, TypeError, ValueError):
        return False


def load_valid_checkpoint(
    path: Path,
    config: SolverConfig,
    index: int,
    wi: float,
    beta: float,
    verify_constraints: bool = True,
) -> Checkpoint | None:
    """Load only a fully written, numerically compatible checkpoint."""
    if not path.exists():
        return None
    required = {
        "schema_version", "config_version", "config_hash", "config_json", "complete",
        "index", "wi", "beta", "z2_fixed", "anchor_z3", "anchor_profiles",
        "anchor_objective", "anchor_residual", "anchor_success", "anchor_nit",
        "anchor_candidate_count", "anchor_retry", "anchor_message", "center_profile",
        "center_objective", "center_residual", "center_success", "raw_center_extent",
        "center_extent", "edge_lo", "edge_hi", "above_onset", "elapsed_seconds",
    }
    try:
        with np.load(path, allow_pickle=False) as data:
            if not required.issubset(data.files):
                return None
            if _scalar_text(data["schema_version"]) != SCHEMA_VERSION:
                return None
            if int(np.asarray(data["config_version"]).item()) != CONFIG_VERSION:
                return None
            if _scalar_text(data["config_hash"]) != config.fingerprint:
                return None
            if _scalar_text(data["config_json"]) != config.config_json:
                return None
            if not bool(np.asarray(data["complete"]).item()):
                return None
            if int(np.asarray(data["index"]).item()) != index:
                return None
            if not _require_scalar_close(data, "wi", wi, 1.0e-12):
                return None
            if not _require_scalar_close(data, "beta", beta, 1.0e-10):
                return None
            if not _require_scalar_close(data, "z2_fixed", Z2_FIXED, 1.0e-15):
                return None
            anchors = np.asarray(data["anchor_z3"], dtype=float)
            profiles = np.asarray(data["anchor_profiles"], dtype=float)
            objective = np.asarray(data["anchor_objective"], dtype=float)
            residual = np.asarray(data["anchor_residual"], dtype=float)
            success = np.asarray(data["anchor_success"], dtype=bool)
            nit = np.asarray(data["anchor_nit"], dtype=int)
            candidate_count = np.asarray(data["anchor_candidate_count"], dtype=int)
            retry = np.asarray(data["anchor_retry"], dtype=bool)
            message = np.asarray(data["anchor_message"], dtype="<U400")
            center = np.asarray(data["center_profile"], dtype=float)
            center_objective = float(np.asarray(data["center_objective"]).item())
            center_residual = float(np.asarray(data["center_residual"]).item())
            center_success = bool(np.asarray(data["center_success"]).item())
            raw_extent = float(np.asarray(data["raw_center_extent"]).item())
            extent = float(np.asarray(data["center_extent"]).item())
            edge_lo = float(np.asarray(data["edge_lo"]).item())
            edge_hi = float(np.asarray(data["edge_hi"]).item())
            above = bool(np.asarray(data["above_onset"]).item())
            elapsed = float(np.asarray(data["elapsed_seconds"]).item())
    except (OSError, ValueError, KeyError, EOFError, zipfile.BadZipFile):
        return None

    k = len(config.anchor_z3)
    if (
        not np.array_equal(anchors, config.anchor_z3)
        or profiles.shape != (k, config.n_steps)
        or objective.shape != (k,)
        or residual.shape != (k,)
        or success.shape != (k,)
        or nit.shape != (k,)
        or candidate_count.shape != (k,)
        or retry.shape != (k,)
        or message.shape != (k,)
        or center.shape != (config.n_steps,)
    ):
        return None
    if (
        not np.all(np.isfinite(profiles))
        or not np.all(np.isfinite(objective))
        or not np.all(np.isfinite(residual))
        or not np.all(success)
        or np.any(residual > CONSTRAINT_TOL)
        or np.any(np.abs(profiles) > config.v_bound + 1.0e-10)
        or not np.allclose(center, profiles[0], rtol=0.0, atol=1.0e-13)
        or not (np.isfinite(center_objective) and np.isfinite(center_residual) and center_success)
        or center_residual > CONSTRAINT_TOL
        or not np.isfinite(elapsed)
        or elapsed < 0.0
    ):
        return None
    if verify_constraints:
        h = config.h
        midpoints = (np.arange(config.n_steps, dtype=float) + 0.5) / config.n_steps
        matrix = np.vstack((np.full(config.n_steps, h), h * (1.0 - midpoints)))
        targets = np.column_stack((np.zeros(k), -0.5 * config.anchor_z3))
        actual = profiles @ matrix.T
        if np.max(np.abs(actual - targets)) > CONSTRAINT_TOL:
            return None

    raw_recomputed = _center_extent_from_profile(center, config.h)
    should_be_above = bool(beta > BETA_PLATEAU_ONSET)
    expected_extent = raw_recomputed if should_be_above else 0.0
    if (
        not np.isfinite(raw_extent)
        or not np.isclose(raw_extent, raw_recomputed, rtol=0.0, atol=2.0e-11)
        or above != should_be_above
        or not np.isclose(extent, expected_extent, rtol=0.0, atol=2.0e-11)
        or not np.isclose(edge_lo, -expected_extent, rtol=0.0, atol=2.0e-11)
        or not np.isclose(edge_hi, expected_extent, rtol=0.0, atol=2.0e-11)
    ):
        return None

    return Checkpoint(
        index=index,
        wi=wi,
        beta=beta,
        frame=FrameResult(
            anchor_profiles=profiles,
            anchor_objective=objective,
            anchor_residual=residual,
            anchor_success=success,
            anchor_nit=nit,
            anchor_candidate_count=candidate_count,
            anchor_retry=retry,
            anchor_message=message,
            center_profile=center,
            center_objective=center_objective,
            center_residual=center_residual,
            center_success=center_success,
            raw_center_extent=raw_extent,
            center_extent=extent,
            edge_lo=edge_lo,
            edge_hi=edge_hi,
            above_onset=above,
            elapsed_seconds=elapsed,
        ),
    )


def _center_extent_from_profile(profile: np.ndarray, h: float) -> float:
    path = np.zeros(len(profile) + 1, dtype=float)
    path[1:] = h * np.cumsum(profile)
    mean = h * (0.5 * path[0] + np.sum(path[1:-1]) + 0.5 * path[-1])
    return float(2.0 * np.max(np.abs(path - mean)))


def save_checkpoint(path: Path, config: SolverConfig, index: int, wi: float, beta: float, frame: FrameResult) -> None:
    """Atomically save one certified frame, including its full compatibility key."""
    atomic_savez(
        path,
        schema_version=np.asarray(SCHEMA_VERSION),
        config_version=np.asarray(CONFIG_VERSION, dtype=np.int64),
        config_hash=np.asarray(config.fingerprint),
        config_json=np.asarray(config.config_json),
        complete=np.asarray(True),
        saved_utc=np.asarray(datetime.now(UTC).isoformat()),
        index=np.asarray(index, dtype=np.int64),
        wi=np.asarray(wi, dtype=float),
        beta=np.asarray(beta, dtype=float),
        z2_fixed=np.asarray(Z2_FIXED, dtype=float),
        anchor_z3=config.anchor_z3,
        anchor_profiles=frame.anchor_profiles,
        anchor_profile_kind=np.asarray("continuum_slope_v=m_prime"),
        anchor_objective=frame.anchor_objective,
        anchor_residual=frame.anchor_residual,
        anchor_success=frame.anchor_success,
        anchor_nit=frame.anchor_nit,
        anchor_candidate_count=frame.anchor_candidate_count,
        anchor_retry=frame.anchor_retry,
        anchor_message=frame.anchor_message,
        center_profile=frame.center_profile,
        center_objective=np.asarray(frame.center_objective),
        center_residual=np.asarray(frame.center_residual),
        center_success=np.asarray(frame.center_success),
        raw_center_extent=np.asarray(frame.raw_center_extent),
        center_extent=np.asarray(frame.center_extent),
        edge_lo=np.asarray(frame.edge_lo),
        edge_hi=np.asarray(frame.edge_hi),
        above_onset=np.asarray(frame.above_onset),
        elapsed_seconds=np.asarray(frame.elapsed_seconds),
    )


def save_failure(path: Path, config: SolverConfig, index: int, wi: float, beta: float, error: BaseException) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "config_version": CONFIG_VERSION,
        "config_hash": config.fingerprint,
        "index": index,
        "wi": wi,
        "beta": beta,
        "saved_utc": datetime.now(UTC).isoformat(),
        "error": str(error),
        "traceback": traceback.format_exc(),
    }
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def write_aggregate(output_root: Path, config: SolverConfig, wi: np.ndarray, completed: dict[int, Checkpoint]) -> Path:
    """Publish a complete or partial aggregate without trusting stale files."""
    n_frames = len(wi)
    k = len(config.anchor_z3)
    n = config.n_steps
    anchor_profiles = np.full((n_frames, k, n), np.nan, dtype=float)
    anchor_objective = np.full((n_frames, k), np.nan, dtype=float)
    anchor_residual = np.full((n_frames, k), np.nan, dtype=float)
    anchor_success = np.zeros((n_frames, k), dtype=bool)
    anchor_nit = np.full((n_frames, k), -1, dtype=int)
    anchor_candidate_count = np.zeros((n_frames, k), dtype=int)
    anchor_retry = np.zeros((n_frames, k), dtype=bool)
    center_profile = np.full((n_frames, n), np.nan, dtype=float)
    center_objective = np.full(n_frames, np.nan, dtype=float)
    center_residual = np.full(n_frames, np.nan, dtype=float)
    center_success = np.zeros(n_frames, dtype=bool)
    raw_center_extent = np.full(n_frames, np.nan, dtype=float)
    center_extent = np.full(n_frames, np.nan, dtype=float)
    edge_lo = np.full(n_frames, np.nan, dtype=float)
    edge_hi = np.full(n_frames, np.nan, dtype=float)
    above_onset = np.zeros(n_frames, dtype=bool)
    elapsed = np.full(n_frames, np.nan, dtype=float)
    complete = np.zeros(n_frames, dtype=bool)

    for index, checkpoint in completed.items():
        frame = checkpoint.frame
        anchor_profiles[index] = frame.anchor_profiles
        anchor_objective[index] = frame.anchor_objective
        anchor_residual[index] = frame.anchor_residual
        anchor_success[index] = frame.anchor_success
        anchor_nit[index] = frame.anchor_nit
        anchor_candidate_count[index] = frame.anchor_candidate_count
        anchor_retry[index] = frame.anchor_retry
        center_profile[index] = frame.center_profile
        center_objective[index] = frame.center_objective
        center_residual[index] = frame.center_residual
        center_success[index] = frame.center_success
        raw_center_extent[index] = frame.raw_center_extent
        center_extent[index] = frame.center_extent
        edge_lo[index] = frame.edge_lo
        edge_hi[index] = frame.edge_hi
        above_onset[index] = frame.above_onset
        elapsed[index] = frame.elapsed_seconds
        complete[index] = True

    output = output_root / "ninf_joint_z3_wi_001_10.npz"
    atomic_savez(
        output,
        schema_version=np.asarray(SCHEMA_VERSION),
        config_version=np.asarray(CONFIG_VERSION, dtype=np.int64),
        config_hash=np.asarray(config.fingerprint),
        config_json=np.asarray(config.config_json),
        z2_fixed=np.asarray(Z2_FIXED),
        wi_values=wi,
        beta_values=np.asarray(beta_from_wi(wi), dtype=float),
        anchor_z3=config.anchor_z3,
        anchor_profile_kind=np.asarray("continuum_slope_v=m_prime"),
        completed=complete,
        complete=complete,
        all_complete=np.asarray(bool(np.all(complete))),
        anchor_profiles=anchor_profiles,
        anchor_objective=anchor_objective,
        anchor_residual=anchor_residual,
        anchor_success=anchor_success,
        anchor_nit=anchor_nit,
        anchor_candidate_count=anchor_candidate_count,
        anchor_retry=anchor_retry,
        center_profile=center_profile,
        center_objective=center_objective,
        center_residual=center_residual,
        center_success=center_success,
        raw_center_extent=raw_center_extent,
        center_extent=center_extent,
        edge_lo=edge_lo,
        edge_hi=edge_hi,
        maximizer_set_lower=edge_lo,
        maximizer_set_upper=edge_hi,
        above_onset=above_onset,
        beta_plateau_onset=np.asarray(BETA_PLATEAU_ONSET),
        wi_plateau_onset=np.asarray(BETA_PLATEAU_ONSET * WI_PER_BETA),
        elapsed_seconds=elapsed,
        updated_utc=np.asarray(datetime.now(UTC).isoformat()),
    )
    return output


def write_progress(output_root: Path, wi: np.ndarray, completed: dict[int, Checkpoint], failures_dir: Path) -> None:
    lines = ["index\tWi\tbeta\tstatus\telapsed_seconds\tmax_constraint_residual\tedge_lo\tedge_hi"]
    for index, value in enumerate(wi):
        beta = float(beta_from_wi(value))
        checkpoint = completed.get(index)
        if checkpoint is None:
            failure = failures_dir / f"frame_{index:03d}_wi_{value:.6f}.json"
            status = "failed" if failure.exists() else "pending"
            lines.append(f"{index}\t{value:.6f}\t{beta:.12g}\t{status}\t\t\t\t")
            continue
        frame = checkpoint.frame
        lines.append(
            f"{index}\t{value:.6f}\t{beta:.12g}\tcomplete\t{frame.elapsed_seconds:.6f}\t"
            f"{np.max(frame.anchor_residual):.3e}\t{frame.edge_lo:.12g}\t{frame.edge_hi:.12g}"
        )
    atomic_write_text(output_root / "progress.tsv", "\n".join(lines) + "\n")


@dataclass
class LockHandle:
    path: Path
    token: str


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


def acquire_lock(path: Path) -> LockHandle:
    """Acquire an exclusive local-host lock, refusing ambiguous remote locks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    host = socket.gethostname()
    token = uuid.uuid4().hex
    payload = {"pid": os.getpid(), "hostname": host, "token": token, "started_utc": datetime.now(UTC).isoformat()}
    for _ in range(2):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                old = json.loads(path.read_text(encoding="utf-8"))
                old_pid = int(old.get("pid", -1))
                old_host = str(old.get("hostname", ""))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                raise RuntimeError(f"Refusing to replace unreadable runner lock: {path}")
            if old_host and old_host != host:
                raise RuntimeError(f"Runner lock belongs to host {old_host}; refusing to break it: {path}")
            if _process_is_alive(old_pid):
                raise RuntimeError(f"Runner lock is held by live PID {old_pid}: {path}")
            # A same-host stale PID is safe to replace.  It can only remove
            # this explicit narrow lock path, never a user directory.
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            continue
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            return LockHandle(path=path, token=token)
        except BaseException:
            try:
                path.unlink()
            except FileNotFoundError:
                pass
            raise
    raise RuntimeError(f"Could not acquire runner lock after clearing a stale one: {path}")


def release_lock(lock: LockHandle) -> None:
    """Remove only the lock instance created by this process."""
    try:
        current = json.loads(lock.path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return
    if str(current.get("token", "")) == lock.token:
        try:
            lock.path.unlink()
        except FileNotFoundError:
            pass


def _preflight_indices(wi: np.ndarray) -> list[int]:
    """Representative low/onset/tail frames promised by the execution plan."""
    probes = (0.02, 3.78, 3.80, 4.00, 6.00, 8.00, 10.00)
    indices = [int(np.argmin(np.abs(wi - probe))) for probe in probes]
    if not all(np.isclose(wi[index], probe, rtol=0.0, atol=1.0e-12) for index, probe in zip(indices, probes)):
        raise RuntimeError("The fixed Wi grid does not contain every preflight probe.")
    return indices


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--n-steps", type=int, default=int(os.environ.get("N_INF_STEPS", "80")), help="Even path discretization N (default: 80).")
    parser.add_argument("--maxiter", type=int, default=800)
    parser.add_argument("--retry-maxiter", type=int, default=1600)
    parser.add_argument("--ftol", type=float, default=1.0e-10)
    parser.add_argument("--low-beta-starts", type=int, default=3)
    parser.add_argument("--high-beta-starts", type=int, default=10)
    parser.add_argument("--index", type=int, action="append", help="Fixed Wi-grid index to compute (repeatable).")
    parser.add_argument("--max-frames", type=int, help="Limit newly computed selected frames.")
    parser.add_argument("--preflight", action="store_true", help="Select 0.02, 3.78, 3.80, 4, 6, 8, 10 unless --index is supplied.")
    parser.add_argument("--config", action="store_true", help="Print config JSON/fingerprint and exit without taking a lock.")
    parser.add_argument("--force", action="store_true", help="Recompute selected compatible checkpoints.")
    parser.add_argument("--no-aggregate", action="store_true", help="Do not refresh aggregate/progress after frames.")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop after the first failed selected frame.")
    args = parser.parse_args(argv)
    if args.n_steps < 8 or args.n_steps % 2:
        parser.error("--n-steps must be even and at least 8 so the V-path exactly satisfies both constraints.")
    if args.maxiter < 1 or args.retry_maxiter < args.maxiter:
        parser.error("Require 1 <= --maxiter <= --retry-maxiter.")
    if args.low_beta_starts < 1 or args.high_beta_starts < 1:
        parser.error("Multistart counts must be positive.")
    if args.max_frames is not None and args.max_frames < 1:
        parser.error("--max-frames must be positive.")
    return args


def _make_config(args: argparse.Namespace) -> SolverConfig:
    return SolverConfig(
        n_steps=args.n_steps,
        anchor_z3=ANCHOR_Z3.copy(),
        v_table_cap=1.0 - 1.0e-6,
        v_bound=1.0 - 2.0e-6,
        maxiter=args.maxiter,
        retry_maxiter=args.retry_maxiter,
        ftol=args.ftol,
        low_beta_starts=args.low_beta_starts,
        high_beta_starts=args.high_beta_starts,
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    wi = make_wi_grid()
    config = _make_config(args)
    if args.config:
        print(json.dumps({"config_hash": config.fingerprint, "config": config.payload()}, indent=2, sort_keys=True))
        return 0

    output_root = args.output_root.resolve()
    checkpoints_dir = output_root / "checkpoints"
    failures_dir = output_root / "failures"
    state_dir = output_root / "state"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    failures_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    lock = acquire_lock(state_dir / "ninf_joint_z3.lock")
    try:
        atomic_write_text(
            state_dir / "config.json",
            json.dumps({"config_hash": config.fingerprint, "config": config.payload()}, indent=2, sort_keys=True) + "\n",
        )
        atomic_write_text(
            state_dir / "status.txt",
            f"running pid={os.getpid()} started_utc={datetime.now(UTC).isoformat()} config_hash={config.fingerprint}\n",
        )
        completed: dict[int, Checkpoint] = {}
        for index, value in enumerate(wi):
            beta = float(beta_from_wi(value))
            existing = load_valid_checkpoint(
                checkpoint_path(checkpoints_dir, index, float(value)), config, index, float(value), beta
            )
            if existing is not None:
                completed[index] = existing

        if args.index:
            requested = sorted(set(args.index))
            invalid = [index for index in requested if index < 0 or index >= len(wi)]
            if invalid:
                raise SystemExit(f"Wi-grid indices must be 0..{len(wi) - 1}; got {invalid}")
            selected = requested
        elif args.preflight:
            selected = _preflight_indices(wi)
        else:
            selected = list(range(len(wi)))
        pending = [index for index in selected if args.force or index not in completed]
        if args.max_frames is not None:
            pending = pending[: args.max_frames]

        print(
            f"n->infinity joint z3 Wi sweep: {len(wi)} points; Wi={wi[0]:.6g}..{wi[-1]:.6g}; "
            f"beta={beta_from_wi(wi[0]):.8g}..{beta_from_wi(wi[-1]):.8g}",
            flush=True,
        )
        print(
            f"N={config.n_steps}; anchors={config.anchor_z3.tolist()}; onset Wi={BETA_PLATEAU_ONSET * WI_PER_BETA:.9g}; "
            f"valid={len(completed)}, selected={len(selected)}, pending={len(pending)}; config={config.fingerprint[:12]}",
            flush=True,
        )
        if not pending:
            if not args.no_aggregate:
                aggregate = write_aggregate(output_root, config, wi, completed)
                write_progress(output_root, wi, completed, failures_dir)
                print(f"No selected frames need work; aggregate refreshed -> {aggregate}", flush=True)
            atomic_write_text(state_dir / "status.txt", f"idle completed={len(completed)}/{len(wi)} updated_utc={datetime.now(UTC).isoformat()}\n")
            return 0

        print("Building stable scaled-Bessel rate table...", flush=True)
        solver = VariationalSolver(config)
        print("Rate table ready.", flush=True)
        failures: list[int] = []
        for ordinal, index in enumerate(pending, start=1):
            value = float(wi[index])
            beta = float(beta_from_wi(value))
            prior: np.ndarray | None = None
            for previous_index in range(index - 1, -1, -1):
                checkpoint = completed.get(previous_index)
                if checkpoint is not None:
                    prior = checkpoint.frame.anchor_profiles
                    break
            print(f"[{ordinal}/{len(pending)}] index={index:03d} Wi={value:.6f} beta={beta:.12g}", flush=True)
            try:
                frame = solver.solve_frame(beta, prior_anchors=prior)
                target = checkpoint_path(checkpoints_dir, index, value)
                save_checkpoint(target, config, index, value, beta, frame)
                completed[index] = Checkpoint(index=index, wi=value, beta=beta, frame=frame)
                # Do not let an old failure label survive a later successful retry.
                try:
                    (failures_dir / f"frame_{index:03d}_wi_{value:.6f}.json").unlink()
                except FileNotFoundError:
                    pass
                print(
                    f"  complete {frame.elapsed_seconds:.2f}s; max residual={np.max(frame.anchor_residual):.2e}; "
                    f"set=[{frame.edge_lo:.8g},{frame.edge_hi:.8g}] -> {target}",
                    flush=True,
                )
                if not args.no_aggregate:
                    aggregate = write_aggregate(output_root, config, wi, completed)
                    write_progress(output_root, wi, completed, failures_dir)
                    print(f"  aggregate {len(completed)}/{len(wi)} -> {aggregate}", flush=True)
            except Exception as error:
                failures.append(index)
                failure = failures_dir / f"frame_{index:03d}_wi_{value:.6f}.json"
                save_failure(failure, config, index, value, beta, error)
                print(f"  FAILED: {error}; diagnostics -> {failure}", file=sys.stderr, flush=True)
                if args.stop_on_error:
                    break

        if not args.no_aggregate:
            aggregate = write_aggregate(output_root, config, wi, completed)
            write_progress(output_root, wi, completed, failures_dir)
            print(f"Final aggregate refresh -> {aggregate}", flush=True)
        state = "failed" if failures else "idle"
        atomic_write_text(
            state_dir / "status.txt",
            f"{state} completed={len(completed)}/{len(wi)} failures={failures} updated_utc={datetime.now(UTC).isoformat()}\n",
        )
        if failures:
            print(f"{len(failures)} frame(s) failed and remain resumable: {failures}", file=sys.stderr)
            return 1
        return 0
    finally:
        release_lock(lock)


if __name__ == "__main__":
    raise SystemExit(main())
