# Single-Molecule Experiments and Image-Based Fold Coordinate

This folder, together with the shared image code in `../phase_trans_from_image/`, reproduces the experimental and image-based results of the paper:

| Paper result | Script | Output |
|---|---|---|
| Fig. 3(a), definition of the image coordinate | `code/fig_S1a.py` | `figures/supplement/fig_S1a_Z3tilde.{png,pdf}` |
| Fig. 3(c), conditional odd response at 10/20/40/50 V | `code/fig_exp_normalform.py` | `figures/main/fig_exp_normalform.{png,pdf,svg}` and the two `_inset_{20V,40V}` files |
| Sec. S4.B table (N, lag pairs, r_L), bootstrap, 108 robustness settings | `code/analysis_stats.py` (printed) | `results/stats_curves.npz`, `results/stats_slope40.npz`, `results/stats_robust40.csv` |
| Fig. S2 | `code/fig_S_combined.py` | `figures/supplement/fig_S_combined.{png,pdf}` |
| Fig. 5, image coordinate on rendered simulation frames | `code/fig_S8.py` | `figures/supplement/fig_S8_proxy_fidelity.{png,pdf}`, `results/stats_proxy.npz` |

Fig. 3(b) is a schematic. Axis labels, panel titles and voltage labels of Figs. 3(a) and 3(c) are added by hand; the scripts draw tick labels only.

## Data

Raw experimental frames are the Hugging Face dataset `MLDS-NUS/polymer-exp-voltage-frames`; the scripts read them from `FinalCODE/data/exp_data/voltage-frames/` (download command in `../README.md`). They are needed for Fig. 3(a) and for the feature extraction only:

- `10V/2026-07-15_1722/`: 48 video tars, the 10 V ensemble; it is also its own image-scale reference `L_max`.
- `20V_2_Feb25.pkl`, `40V_Feb25.pkl`, `50V_Feb25.pkl`: the 20, 40 and 50 V ensembles; 50 V is also the `L_max` reference for 20-50 V (same imaging batch).

Fig. 5 uses image features of rendered simulation frames, written to `FinalCODE/data/sim_cache/`, and the physical observables of the machine-learning pipeline (`../examples/UIDD_polymer_dynamics_wi/z2-z3-Interpret/cache/observables_stride50.npz`, included). The rendered features are not included; they are rendered from the raw simulation dataset `MLDS-NUS/Wis_ParamSweep_New`, see "Simulated Images" below.

## Setup

CPU only. Any environment from the repository works; `numpy>=2` is required to unpickle the `.pkl` frames.

```bash
cd /path/to/FinalCODE  # LOCAL: project directory
conda env create -f environment.cpu.yml  # skip if it already exists
conda activate phase-diagram-cpu
export MPLBACKEND=Agg
```

All scripts locate their inputs relative to their own file, so they can be run from any directory. Run the blocks below in order from `FinalCODE`.

## Experiments: Figs. 3 and S2

All outputs of this block, the feature caches in `data/exp_data/cache_voltage_frames/`, `results/` and the figures, are included. The two extraction scripts skip every video or trajectory whose cache file already exists; delete `data/exp_data/cache_voltage_frames/` first to extract again from the raw frames.

```bash
# Per-frame image features -> data/exp_data/cache_voltage_frames/ (a few minutes)
python -B phase_trans_from_image/exp_data_image/extract_features.py      # 48 video tars
python -B phase_trans_from_image/exp_data_image/extract_pkl_batches.py   # three pickles

# Output folders are not created by the scripts
mkdir -p phase_trans_from_image_2_z3/results \
         phase_trans_from_image_2_z3/figures/main \
         phase_trans_from_image_2_z3/figures/supplement

# Trajectory bootstrap (2000 resamples) and robustness grid, about 1 min
python -B phase_trans_from_image_2_z3/code/analysis_stats.py

python -B phase_trans_from_image_2_z3/code/fig_exp_normalform.py   # Fig. 3(c)
python -B phase_trans_from_image_2_z3/code/fig_S_combined.py       # Fig. S2
python -B phase_trans_from_image_2_z3/code/fig_S1a.py              # Fig. 3(a); loads the 5.6 GB 40 V pickle
```

`analysis_stats.py` prints the values quoted in Sec. S4.B: the lag pair, `r_L` and the number of contributing trajectories per voltage, and the robustness counts (72 of 108 settings admissible, all with `s0 < 0` at 20 V and `s0 > 0` at 40 V, outer zero found in 54 of 72).

## Simulated Images: Fig. 5

Every simulation frame is rendered to a microscope image (Gibson-Lanni PSF from `../Langevin_single_chain/microscPSF.py`, shot noise, 8 bit) and measured with the same per-frame function as the experimental frames.

The figure and its statistics (`figures/supplement/fig_S8_proxy_fidelity.{png,pdf}`, `results/stats_proxy.npz`) are included; the rendered features are not.

**To reproduce Fig. 5, first run `load_dataset("MLDS-NUS/Wis_ParamSweep_New")` once.** It downloads the parquet files that are rendered here (about 190 GB). The physical observables are included; recomputing them as well uses the Arrow cache that the same call builds (see `../examples/UIDD_polymer_dynamics_wi/README.md`, "Polymer Traces and Physical Observables").

`LOCAL`: set `HF_GLOB` in `phase_trans_from_image/simulation_data_image/extract_features.py` to `<HF_HOME>/hub/datasets--MLDS-NUS--{repo}/snapshots/*/data/*.parquet`. `NPROC` (default 30) sets the number of worker processes.

```bash
export HF_HOME=/path/to/huggingface_cache  # LOCAL: download directory
python -c 'from datasets import load_dataset; load_dataset("MLDS-NUS/Wis_ParamSweep_New")'

# Builds the PSF stack and calibrates the gain -> data/sim_cache/render_calib.npz, then
# renders 11,145 trajectories -> data/sim_cache/imgfeat_all/ (about 400 MB, about 45 min
# with 30 processes). A rerun skips the calibration and every finished shard.
python -B phase_trans_from_image/simulation_data_image/extract_features.py paramsweep_new
# Needs examples/UIDD_polymer_dynamics_wi/z2-z3-Interpret/cache/observables_stride50.npz (included)
python -B phase_trans_from_image_2_z3/code/fig_S8.py
```

`fig_S8.py` prints the per-Wi correlations shown in Fig. 5 (r = 0.543, 0.571, 0.509, 0.630).

## Notes

- Intermediate data (`data/exp_data/cache_voltage_frames/`, `results/`) and all figures are included; the commands above regenerate them, after the deletions noted there. The rendered simulation features (`data/sim_cache/`) are not included.
- `code/verify.py` holds only the experimental loader `load_exp` used by `analysis_stats.py`.
- `simulation_data_image/extract_features.py` also accepts `holdout_new`, `sweep_old`, `changebead` and `truth`, and `render.py` with no argument writes a demo image. None of these are needed for the paper.
- Regenerated feature caches match the original ones exactly in every field used by the analysis. The rendered features depend on the calibrated gain, which can differ in the sixth digit; the Fig. 5 correlations then agree to about 1e-4.
