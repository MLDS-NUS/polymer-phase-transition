# Analytical Model: Long-Chain Limit and Finite-n Monte Carlo

This folder reproduces the analytical-model results of the paper: 

## Setup

CPU only; numpy, scipy and matplotlib are required.

```bash
cd /path/to/FinalCODE/FreeEnergybyIntegration  # LOCAL: project directory
conda activate phase-diagram-cpu                # created from ../environment.cpu.yml
export MPLBACKEND=Agg OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
```

All scripts locate their inputs relative to their own file. Results are written below `data/`, `mc/` and `ninf/` next to each script; these results and all figures are included. Every runner is resumable: a rerun skips results whose configuration matches and that are complete, so with the included results the commands below finish at once. To recompute, delete `1-trans-z2/wi_001_10/data/`, `2-trans-z3/wi_001_10/data/`, `1-trans-z2/fixed_points/mc/`, `2-trans-z3/fixed_points/mc/` and `2-trans-z3/profile_targets/ninf/` first.

## 1. Long-Chain Limit (n -> inf)

```bash
python -u 1-trans-z2/wi_001_10/run_ninf_wi_001_10.py            # 141 Wi values
python -u 2-trans-z3/wi_001_10/run_ninf_joint_z3_wi_001_10.py    # 340 Wi values
```

Outputs: `1-trans-z2/wi_001_10/data/ninf/ninf_wi_001_10.npz` and `2-trans-z3/wi_001_10/data/ninf/ninf_joint_z3_wi_001_10.npz`. Each takes on the order of an hour on one core; the z3 sweep is the longer one.

## 2. Finite-n Monte Carlo at the Figure Points

The figure needs four Wi values only, and the fixed-point runners compute exactly those. Each one uses the estimator, grids, kernel widths, sample counts and quality checks of the original Wi sweeps, and writes one file `fixed_points/mc/n<n>_wi_<Wi>.npz` per (n, Wi). Wi need not lie on the sweep grid.

```bash
for n in 8 16 32; do
  # uniform proposal, M = 1e9; 2-8 min per point with 6 workers
  python -u 1-trans-z2/fixed_points/run_mc_fixed.py --n "$n" --wi 0.02 1.32 --workers 6
  # tilted proposal, M = 2e7 raised up to 1e8 until the quality check passes; 1-3 min per point
  python -u 2-trans-z3/fixed_points/run_mc_tilted_fixed.py --n "$n" --wi 1.32 5.28 --workers 4
done

# n -> inf section on the Monte Carlo z3 grid at Wi = 1.32 and 5.28; needs the z3 sweep of step 1
python -u 2-trans-z3/profile_targets/run_ninf_profile_wi_0p66_2p64.py
```

The tilted runner solves its six continuum anchor profiles itself, so it does not need step 1. Its proposal mixes the anchors, their cyclic shifts and their x-reflections; the reflections cover the z3 < 0 half, which the shifts alone leave almost unsampled.

The number of workers is part of the random-stream definition. Keep the values above to obtain the same numbers on a rerun.

Known limitation: for n = 32 at Wi = 1.32 the uniform-proposal estimate of `F_2` is not converged. Independent runs place the minimum one grid cell apart (z2 = 0.55 or 0.65 against 0.60 from annealed SMC). n = 8 and 16 are not affected.

## 3. Figures

```bash
python -B plot_F_grid_panels.py          # Fig. 2(e-h): figs/{F2_Wi0.02,F2_Wi1.32,F23cut_Wi1.32,F23cut_Wi5.28}.{png,pdf,svg}
python -B plot_transition_2x2_panels.py  # Fig. 2(i-n) panels: ../figs/transition_*.{png,pdf,svg}
```

```bash
python -B plot_F_grid_z2_z3.py  
python -B 2-trans-z3/profile_targets/plot_profiles_wi_0p66_2p64.py  
python -B plot_argmax_ninf_combined.py 
python -B plot_transition_2x2.py 
```

Text metrics depend on the matplotlib version; other versions shift some anti-aliased pixels but not the content.

 
