#!/usr/bin/env python3
"""Continuum ``U_infinity(z3)`` on the MC grid at ``Wi=1.32`` and ``Wi=5.28``.

The finished ``wi_001_10`` sweep stores the ``n -> infinity`` objective at only
six ``z3`` anchors per Wi, which is enough for the maximizer-set endpoints but
far too coarse to draw next to the 99-point MC profiles.  This runner reuses
the *same* validated solver (``run_ninf_joint_z3_wi_001_10.VariationalSolver``
with the sweep's ``SolverConfig`` numbers) and simply evaluates it on the MC's
own ``z3`` grid.

Only the non-negative half is solved: for ``z2=0`` the constraint target is
``(0, -z3/2)`` and the Bessel rate is even in the slope, so the objective is
even in ``z3``.  The negative half is the exact mirror, not an interpolation.

Each grid point is written as its own atomic checkpoint, so an interrupted run
resumes without repeating converged work.  After a Wi finishes, the aggregate
``ninf/ninf_profile_wi_<tag>.npz`` is published atomically and the six stored
sweep anchors are re-checked against this run (``anchor_check_max_abs_diff``).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np


HERE = Path(__file__).resolve().parent
SWEEP_DIR = HERE.parent / "wi_001_10"
sys.path.insert(0, str(SWEEP_DIR))

from common import BETA_PER_WI, atomic_savez, atomic_write_text  # noqa: E402
from run_ninf_joint_z3_wi_001_10 import (  # noqa: E402
    BETA_PLATEAU_ONSET,
    SolverConfig,
    VariationalSolver,
)

TARGET_WI = (1.32, 5.28)
SWEEP_NINF = SWEEP_DIR / "data" / "ninf" / "ninf_joint_z3_wi_001_10.npz"
OUTPUT_ROOT = HERE / "ninf"
# The sweep's own numerical settings; changing any of these makes the fine
# grid incomparable with the stored six-anchor results it is checked against.
N_STEPS = 80
V_TABLE_CAP = 1.0 - 1.0e-6
V_BOUND = 1.0 - 2.0e-6
MAXITER = 800
RETRY_MAXITER = 1600
FTOL = 1.0e-10
LOW_BETA_STARTS = 3
HIGH_BETA_STARTS = 10
ANCHOR_CHECK_TOL = 1.0e-6


def wi_tag(wi: float) -> str:
    return f"{wi:.2f}".replace(".", "p")


def mc_half_grid() -> np.ndarray:
    """Return the non-negative half of the MC z3 grid, verified symmetric."""
    from run_mc_tilted_wi_001_10 import Z3_GRID
    z3 = np.asarray(Z3_GRID, dtype=float)
    if not np.allclose(z3, -z3[::-1], rtol=0.0, atol=1e-12):
        raise ValueError("the MC z3 grid is not mirror symmetric")
    half = z3[z3 >= 0.0]
    if half.size * 2 - 1 != z3.size:
        raise ValueError("the MC z3 grid has no exact zero midpoint")
    return half


def make_solver(anchor_z3: np.ndarray) -> VariationalSolver:
    config = SolverConfig(
        n_steps=N_STEPS,
        anchor_z3=np.asarray(anchor_z3, dtype=float),
        v_table_cap=V_TABLE_CAP,
        v_bound=V_BOUND,
        maxiter=MAXITER,
        retry_maxiter=RETRY_MAXITER,
        ftol=FTOL,
        low_beta_starts=LOW_BETA_STARTS,
        high_beta_starts=HIGH_BETA_STARTS,
    )
    return VariationalSolver(config)


def sweep_anchor_profiles(wi: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (anchor_z3, anchor_objective, anchor_profiles) from the sweep."""
    with np.load(SWEEP_NINF, allow_pickle=False) as data:
        wi_values = np.asarray(data["wi_values"], dtype=float)
        index = int(np.argmin(np.abs(wi_values - wi)))
        if not np.isclose(wi_values[index], wi, rtol=0.0, atol=5e-13):
            raise ValueError(f"Wi={wi:g} is absent from the sweep grid")
        if not bool(np.asarray(data["completed"])[index]):
            raise ValueError(f"Sweep Wi={wi:g} is not complete")
        return (
            np.asarray(data["anchor_z3"], dtype=float),
            np.asarray(data["anchor_objective"], dtype=float)[index],
            np.asarray(data["anchor_profiles"], dtype=float)[index],
        )


def warm_start_for(z3: float, anchor_z3: np.ndarray,
                   anchor_profiles: np.ndarray) -> np.ndarray:
    """Nearest stored sweep anchor profile, used as an extra multistart seed."""
    return anchor_profiles[int(np.argmin(np.abs(anchor_z3 - z3)))].copy()


def point_path(checkpoint_dir: Path, wi: float, index: int, z3: float) -> Path:
    return checkpoint_dir / f"{wi_tag(wi)}_pt_{index:03d}_z3_{z3:.6f}.npz"


def load_point(path: Path, beta: float, z3: float) -> dict | None:
    """Return a reusable checkpoint, or None when absent/incompatible."""
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if not bool(np.asarray(data["complete"]).item()):
                return None
            if int(np.asarray(data["n_steps"]).item()) != N_STEPS:
                return None
            for key, expected in (("beta", beta), ("z3", z3)):
                if not np.isclose(float(np.asarray(data[key]).item()), expected,
                                  rtol=0.0, atol=1e-12):
                    return None
            profile = np.asarray(data["profile"], dtype=float)
            objective = float(np.asarray(data["objective"]).item())
            residual = float(np.asarray(data["residual"]).item())
    except (KeyError, OSError, ValueError):
        return None
    if profile.shape != (N_STEPS,) or not np.isfinite(profile).all():
        return None
    if not np.isfinite(objective):
        return None
    return dict(profile=profile, objective=objective, residual=residual)


def solve_one_wi(wi: float, *, force: bool, verbose: bool) -> Path:
    beta = float(wi) * BETA_PER_WI
    half = mc_half_grid()
    anchor_z3, anchor_objective, anchor_profiles = sweep_anchor_profiles(wi)
    solver = make_solver(half)

    checkpoint_dir = OUTPUT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    objective = np.full(half.size, np.nan, dtype=float)
    residual = np.full(half.size, np.nan, dtype=float)
    profiles = np.full((half.size, N_STEPS), np.nan, dtype=float)
    reused = np.zeros(half.size, dtype=bool)

    started = perf_counter()
    previous: np.ndarray | None = None
    for index, z3 in enumerate(half):
        z3 = float(z3)
        path = point_path(checkpoint_dir, wi, index, z3)
        cached = None if force else load_point(path, beta, z3)
        if cached is None:
            best, _ = solver.solve_anchor(
                z3, beta, previous,
                warm_start_for(z3, anchor_z3, anchor_profiles),
            )
            if not best.success or not np.isfinite(best.value):
                raise RuntimeError(f"Wi={wi:g}, z3={z3:g} did not converge")
            atomic_savez(
                path,
                complete=np.array(True),
                wi=np.array(wi, dtype=float),
                beta=np.array(beta, dtype=float),
                z3=np.array(z3, dtype=float),
                n_steps=np.array(N_STEPS, dtype=np.int64),
                profile=best.v,
                objective=np.array(best.value, dtype=float),
                residual=np.array(best.residual, dtype=float),
                source=np.array(best.source),
            )
            cached = dict(profile=best.v, objective=best.value,
                          residual=best.residual)
        else:
            reused[index] = True
        objective[index] = cached["objective"]
        residual[index] = cached["residual"]
        profiles[index] = cached["profile"]
        previous = cached["profile"].copy()
        if verbose:
            print(f"  Wi={wi:.2f} z3={z3:+.2f}  J={objective[index]:+.10g}"
                  f"{'  (cached)' if reused[index] else ''}", flush=True)

    # Mirror onto the full MC grid; the objective is even in z3 by symmetry.
    z3_full = np.concatenate((-half[:0:-1], half))
    objective_full = np.concatenate((objective[:0:-1], objective))
    potential_full = objective_full.max() - objective_full

    # Four of the six sweep anchors (0.098, 0.196, 0.294, 0.392) are not on the
    # MC grid, so the check re-solves them here rather than interpolating: this
    # verifies that the same solver, warm started along the fine grid, still
    # reproduces the published sweep values.
    check = np.empty(anchor_z3.size, dtype=float)
    for position, value in enumerate(anchor_z3):
        value = float(value)
        neighbour = profiles[int(np.argmin(np.abs(half - value)))]
        best, _ = solver.solve_anchor(value, beta, neighbour, neighbour)
        if not best.success or not np.isfinite(best.value):
            raise RuntimeError(f"Wi={wi:g}: anchor check at z3={value:g} failed")
        check[position] = best.value
    anchor_diff = float(np.max(np.abs(check - anchor_objective)))
    if anchor_diff > ANCHOR_CHECK_TOL:
        raise RuntimeError(
            f"Wi={wi:g}: fine grid disagrees with the stored sweep anchors by "
            f"{anchor_diff:.3g} (> {ANCHOR_CHECK_TOL:g})"
        )

    output = OUTPUT_ROOT / f"ninf_profile_wi_{wi_tag(wi)}.npz"
    atomic_savez(
        output,
        complete=np.array(True),
        wi=np.array(wi, dtype=float),
        beta=np.array(beta, dtype=float),
        above_onset=np.array(beta > BETA_PLATEAU_ONSET),
        beta_plateau_onset=np.array(BETA_PLATEAU_ONSET, dtype=float),
        n_steps=np.array(N_STEPS, dtype=np.int64),
        z3_half=half,
        objective_half=objective,
        residual_half=residual,
        profiles_half=profiles,
        z3_grid=z3_full,
        objective=objective_full,
        potential=potential_full,
        sweep_anchor_z3=anchor_z3,
        sweep_anchor_objective=anchor_objective,
        anchor_check_objective=check,
        anchor_check_max_abs_diff=np.array(anchor_diff, dtype=float),
        elapsed_seconds=np.array(perf_counter() - started, dtype=float),
        reused_points=np.array(int(reused.sum()), dtype=np.int64),
    )
    atomic_write_text(
        HERE / "state" / f"ninf_wi_{wi_tag(wi)}.status",
        f"complete\twi={wi:.2f}\tbeta={beta:.10g}\tpoints={half.size}\t"
        f"reused={int(reused.sum())}\tanchor_diff={anchor_diff:.3g}\n",
    )
    print(f"  -> {output}  (anchor check {anchor_diff:.3g}, "
          f"{perf_counter() - started:.1f}s)", flush=True)
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wi", type=float, action="append",
                        help="Target Wi (repeatable); default: 1.32 and 5.28.")
    parser.add_argument("--force", action="store_true",
                        help="Recompute points even when a checkpoint exists.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    targets = tuple(args.wi) if args.wi else TARGET_WI
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (HERE / "state").mkdir(parents=True, exist_ok=True)
    for wi in targets:
        print(f"Wi={wi:.2f} (beta={wi * BETA_PER_WI:.10g})", flush=True)
        solve_one_wi(float(wi), force=args.force, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
