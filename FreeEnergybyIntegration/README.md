# Analytical Model: Long-Chain Limit and Finite-n Monte Carlo

This folder reproduces the analytical-model results of the paper:

| Paper result | Computed by | Plotted by |
|---|---|---|
| Fig. 2(e,f), `F_{2,n}(z2)` at Wi = 0.02 and 1.32 for n = 8, 16, 32 and n -> inf | `1-trans-z2/` | `plot_F_grid_panels.py` |
| Fig. 2(g,h), `F_{23,n}(0, z3)` at Wi = 1.32 and 5.28 for n = 8, 16, 32 and n -> inf | `2-trans-z3/` | `plot_F_grid_panels.py` |
| Fig. 2(k,n), minimizing sets of `F_{2,inf}` and `F_{23,inf}(0, .)`, Wi_t = 0.95 and 3.79 | the two n -> inf sweeps | `plot_transition_2x2_panels.py` |

**Wi convention.** Every script uses `Wi = beta * 4321 / 45000`, the paper's Wi = 0.096 lambda. The n -> inf sweeps cover Wi = 0.02 to 10: 141 values for `F_2` and 340 for `F_23(0, .)`. Directory names such as `wi_001_10` and file names such as `*_wi_0p66_2p64.py` keep earlier labels; the Wi values inside the scripts are the ones above.

The plotting scripts take their figure geometry and fonts from the machine-learning figures, so run the "Free Energy and Transitions" block of `../examples/UIDD_polymer_dynamics_wi/README.md` first. It writes `2-trans-z3/F_z2_z3_grid_seed1.png`, `1-trans-z2/z2_argmin_data.npz` and `2-trans-z3/z3_argmin_data.npz` there, which are read here.

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

`plot_transition_2x2_panels.py` prints `z2: Wi_t = 0.947598` and `z3: Wi_t = 3.790805`. The `_learned` panels are Fig. 2(i,j,l,m) and the `_ninf` panels are Fig. 2(k,n). With `--ninf-only` it writes only the two theory panels, but it still reads the machine-learning data to size them. Axis labels and the "Wi = ..." titles are added by hand.

Optional figures, not used in the paper:

```bash
python -B plot_F_grid_z2_z3.py           # F_z2_z3_grid.{png,pdf,svg,csv}: the four profiles in one grid
python -B 2-trans-z3/profile_targets/plot_profiles_wi_0p66_2p64.py   # 2-trans-z3/F_z3_wi_1p32_5p28.{png,csv}
python -B plot_argmax_ninf_combined.py   # argmax_ninf_combined.{png,pdf}: both n -> inf minimizing sets
python -B plot_transition_2x2.py         # transition_2x2.{png,pdf}: learned and theoretical transitions
```

Text metrics depend on the matplotlib version; other versions shift some anti-aliased pixels but not the content.

## Code Layout

Only code used for the paper is included. Some modules are needed only for their functions: `2-trans-z3/wi_001_10/run_mc_tilted_wi_001_10.py` supplies the tilted estimator to the fixed-point runner, and `plot_argmax_vs_wi_001_10.py`, `plot_maximizer_vs_wi_001_10.py`, `plot_argmax_ninf_combined.py` and `plot_transition_2x2.py` supply loaders and panel drawing to `plot_transition_2x2_panels.py`. The full-grid Monte Carlo sweeps, their supervisors and preflight checks are not part of this folder.
