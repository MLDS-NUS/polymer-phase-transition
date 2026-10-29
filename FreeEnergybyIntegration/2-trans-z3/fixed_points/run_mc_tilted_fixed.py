#!/usr/bin/env python3
"""Tilted MC of F_23,n(z2=0, z3) at a few fixed Wi only.

The sweep runner ``../wi_001_10/run_mc_tilted_wi_001_10.py`` covers all 340 Wi
values and publishes its aggregate only when every point is done.  The paper
figure needs a handful of Wi values; this runner computes exactly those, with
the same estimator, z3 grid, kernel widths, quality gate and adaptive sample
escalation (M = 2e7 -> 1e8), and publishes one file per (n, Wi):

    mc/n<n>_wi_<Wi:.6f>.npz

The proposal anchors are solved here with the sweep's continuum solver and
numerical settings (six z3 anchors, N = 80), so no n -> inf sweep is needed.
Wi need not lie on the sweep grid.  A finished file is reused on a rerun only
if it was made with the same configuration; ``--force`` recomputes it.

    python -u run_mc_tilted_fixed.py --n 8 --wi 1.32 5.28 --workers 4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SWEEP_DIR = HERE.parent / "wi_001_10"
sys.path.insert(0, str(SWEEP_DIR))

from common import WI_PER_BETA, atomic_savez, beta_from_wi  # noqa: E402
import run_mc_tilted_wi_001_10 as T  # noqa: E402
from run_ninf_joint_z3_wi_001_10 import (  # noqa: E402
    BETA_PLATEAU_ONSET,
    SolverConfig,
    VariationalSolver,
)

OUTPUT_DIR = HERE / "mc"
CONFIG_VERSION = 1
BASE_SEED = 5_301_779
# Continuum anchor solver: the sweep's own numbers (run_ninf_joint_z3 defaults).
SOLVER = dict(n_steps=80, v_table_cap=1.0 - 1.0e-6, v_bound=1.0 - 2.0e-6,
              maxiter=800, retry_maxiter=1600, ftol=1.0e-10,
              low_beta_starts=3, high_beta_starts=10)


def output_path(n: int, wi: float) -> Path:
    return OUTPUT_DIR / f"n{n}_wi_{wi:.6f}.npz"


def run_config(n: int, wi: float, *, workers: int, batch: int, base_m: int, max_m: int) -> dict:
    return {
        "config_version": CONFIG_VERSION,
        "sweep_estimator": T.ESTIMATOR,
        "sweep_config_version": T.CONFIG_VERSION,
        "n": n,
        "wi": float(wi),
        "wi_per_beta": WI_PER_BETA,
        "z2_fixed": T.Z2_FIXED,
        "z3_grid": [float(x) for x in T.Z3_GRID],
        "sigma_z2": T.SIGMA_Z2,
        "sigma_z3": T.SIGMA_Z3,
        "anchor_z3": [float(x) for x in T.ANCHOR_Z3],
        "proposal_slope_cap": T.PROPOSAL_SLOPE_CAP,
        "ess_floor": T.ESS_FLOOR,
        "stability_tol_per_n": T.STABILITY_TOL_PER_N,
        "solver": SOLVER,
        "workers": workers,
        "batch": batch,
        "M_base": base_m,
        "M_max": max_m,
    }


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def is_done(path: Path, digest: str) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return bool(data["complete"]) and str(data["config_hash"]) == digest
    except (OSError, ValueError, KeyError):
        return False


def solve_anchors(beta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The six continuum anchor profiles at this beta, continued in z3."""
    solver = VariationalSolver(SolverConfig(anchor_z3=np.asarray(T.ANCHOR_Z3, float), **SOLVER))
    profiles, objective, residual = [], [], []
    previous = None
    for z3 in T.ANCHOR_Z3:
        best, _ = solver.solve_anchor(float(z3), beta, previous, None)
        if not best.success or not np.isfinite(best.value) or best.residual > 2.0e-7:
            raise RuntimeError(f"anchor z3={z3:g} did not converge at beta={beta:.10g}")
        profiles.append(best.v)
        objective.append(best.value)
        residual.append(best.residual)
        previous = best.v.copy()
    return np.array(profiles), np.array(objective), np.array(residual)


def run_one(n: int, wi: float, args: argparse.Namespace) -> Path:
    beta = float(beta_from_wi(wi))
    config = run_config(n, wi, workers=args.workers, batch=args.batch,
                        base_m=args.M, max_m=args.max_M)
    digest = config_hash(config)
    path = output_path(n, wi)
    if not args.force and is_done(path, digest):
        print(f"n={n} Wi={wi:.6f}: done, reusing {path.name}", flush=True)
        return path

    started = time.time()
    profiles, anchor_objective, anchor_residual = solve_anchors(beta)
    components = T.build_components(n, profiles)
    print(f"n={n} Wi={wi:.6f} beta={beta:.8g}: anchors solved in "
          f"{time.time() - started:.1f}s, {len(components)} proposal components", flush=True)

    m_attempt, attempt = args.M, 0
    while True:
        seed = BASE_SEED + 1_000_003 * n + 104_729 * attempt + int(round(wi * 1e6))
        result = T.tilted_logz(n=n, beta=beta, components=components, m_requested=m_attempt,
                               batch=args.batch, workers=args.workers, seed=seed)
        if not np.isfinite(result["mc_logZ"]).all():
            raise FloatingPointError(f"non-finite tilted MC output at n={n}, Wi={wi:.6f}")
        quality, boundary, ess_min, stable, reason, z_neg, z_pos = T._point_quality(result)
        print(f"  M={m_attempt:,} attempt={attempt}: paired ESS min={ess_min:.1f} "
              f"stable={stable} quality={quality} ({reason}) "
              f"{result['elapsed_seconds']:.0f}s", flush=True)
        if quality or m_attempt >= args.max_M:
            break
        m_attempt, attempt = T._next_m(m_attempt, args.max_M), attempt + 1

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_savez(
        path,
        complete=np.array(True),
        config_hash=np.array(digest),
        config_json=np.array(json.dumps(config, sort_keys=True)),
        n=np.array(n), wi=np.array(wi), beta=np.array(beta),
        above_onset=np.array(beta > BETA_PLATEAU_ONSET),
        z2_fixed=np.array(T.Z2_FIXED), z3_grid=T.Z3_GRID,
        mc_logZ=np.asarray(result["mc_logZ"]), mc_logZ_raw=np.asarray(result["mc_logZ_raw"]),
        ess=np.asarray(result["ess"]), paired_ess=np.asarray(result["paired_ess"]),
        se=np.asarray(result["se"]),
        paired_ess_min=np.array(ess_min), profile_stable=np.array(stable),
        split_profile_disagreement_per_n=np.array(result["split_profile_disagreement_per_n"]),
        quality_pass=np.array(quality), quality_reason=np.array(reason),
        adaptive_exhausted=np.array(bool(not quality and m_attempt >= args.max_M)),
        argmax_at_boundary=np.array(boundary), z_star_neg=np.array(z_neg), z_star_pos=np.array(z_pos),
        rerun_count=np.array(attempt), M_requested=np.array(m_attempt),
        M_actual=np.array(result["M_actual"]), seed=np.array(seed),
        anchor_z3=np.asarray(T.ANCHOR_Z3, float), anchor_profiles=profiles,
        anchor_objective=anchor_objective, anchor_residual=anchor_residual,
        component_count=np.array(len(components)),
        elapsed_seconds=np.array(time.time() - started),
    )
    print(f"  -> {path}", flush=True)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, choices=(8, 16, 32), required=True)
    parser.add_argument("--wi", type=float, nargs="+", required=True, help="Wi values to compute.")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("MC_WORKERS", "4")))
    parser.add_argument("--batch", type=int, default=20_000)
    parser.add_argument("--M", type=int, default=20_000_000, help="Samples of the first attempt.")
    parser.add_argument("--max-M", type=int, default=100_000_000, help="Cap of the adaptive escalation.")
    parser.add_argument("--force", action="store_true", help="Recompute finished points.")
    args = parser.parse_args()
    if args.workers < 2:
        parser.error("--workers must be at least 2 (split-stream stability check)")
    return args


def main() -> None:
    args = parse_args()
    for wi in args.wi:
        run_one(args.n, float(wi), args)


if __name__ == "__main__":
    main()
