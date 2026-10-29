# Hidden Phase Transition in Single-Polymer Stretching under Elongational Flow

Code and data to reproduce the results of the paper. Each analysis folder has its own README with the exact commands; this file describes the environments and what each folder contains.

## Environments

Python 3.12. Two conda environments are provided:

| File | Environment | Used for |
|---|---|---|
| `environment.gpu-cuda12.yml` | `phase-diagram-gpu-cuda12` | `examples/`: training and evaluating the neural-network models (NVIDIA GPU, CUDA 12) |
| `environment.cpu.yml` | `phase-diagram-cpu` | `FreeEnergybyIntegration/`, `phase_trans_from_image/`, `phase_trans_from_image_2_z3/` (CPU only) |

```bash
conda env create -f environment.cpu.yml          # or environment.gpu-cuda12.yml
conda activate phase-diagram-cpu
```

Both environments contain the same packages apart from the JAX build, so the CPU environment also imports `net` and loads the trained models; training and the learned free-energy integrals are practical only on a GPU. `net` is not installed as a package; scripts import it from the repository root by path. `pyproject.toml` and `poetry.lock` describe the same package for Poetry and are not needed with the conda environments. `numpy>=2` is required to unpickle the experimental `.pkl` files.

## Folders

| Folder | Contents |
|---|---|
| `net/` | Library: OnsagerNet and SDE models, dimensionality-reduction transformations, trainers, data augmentation. |
| `examples/utils/` | Dataset and SDE-simulation helpers shared by the example scripts. |
| `examples/UIDD_polymer_dynamics_wi/` | Neural-network pipeline for the polymer data: optional regeneration of the reduced data, training (config in `config/`), the five trained models in `outputs/` (seeds 123, 1, 12, 1234, 12345), the PCA parameters in `data_cache_nolog/`, learned free-energy curves and transitions (`1-trans-z2/`, `2-trans-z3/`), physical interpretation of z2 and z3 (`z2-z3-Interpret/`), trajectory and prediction plots (`plot_polymer/`, `plot_test/`). See its README. |
| `FreeEnergybyIntegration/` | Analytical model: long-chain limit (n → ∞) sweeps over Wi, finite-n Monte Carlo at the figure points, and the free-energy and transition panels of Fig. 2. See its README. |
| `phase_trans_from_image/` | Shared image code: per-frame image features (`frame_features.py`), fold coordinate and frame-quality statistics (`indicators.py`), feature extraction from the experimental videos (`exp_data_image/`) and rendering plus extraction of simulated frames (`simulation_data_image/`). No paper figure is produced here. |
| `phase_trans_from_image_2_z3/` | Experimental analysis: conditional odd response and bootstrap statistics for the 10, 20, 40 and 50 V groups, Figs. 3(a), 3(c), S2 and 5. See its README. |
| `Langevin_single_chain/` | `microscPSF.py`: Gibson–Lanni point-spread function, used to render simulated frames. |
| `figs/` | Fig. 2(i–n) transition panels, written by `FreeEnergybyIntegration/plot_transition_2x2_panels.py`. |
| `data/` | Input data and feature caches, see below. |

## Data

Included in `data/`:

- `reduced_symm_new_train/`, `reduced_symm_new_test/`: reduced simulation datasets (Hugging Face `datasets` format) for training and testing.
- `exp_data/cache_voltage_frames/`: per-frame image features of the experimental videos.

The raw experimental frames are the public Hugging Face dataset [`MLDS-NUS/polymer-exp-voltage-frames`](https://huggingface.co/datasets/MLDS-NUS/polymer-exp-voltage-frames) (about 30 GB): `10V/` (48 video tars of PNG frames) and `20V_2_Feb25.pkl`, `40V_Feb25.pkl`, `50V_Feb25.pkl`. The scripts read them from `data/exp_data/voltage-frames/`; download them there with the `hf` command of either environment:

```bash
hf download MLDS-NUS/polymer-exp-voltage-frames --repo-type dataset --local-dir data/exp_data/voltage-frames
```

They are needed only for Fig. 3(a) and to extract the experimental features again; Figs. 3(c) and S2 and the statistics of Sec. S4.B use the included features.

All intermediate results and figures are included as well: the learned free-energy curves, surfaces, physical observables and prediction statistics in `examples/UIDD_polymer_dynamics_wi/`, the n → ∞ and Monte Carlo results in `FreeEnergybyIntegration/`, the bootstrap and Fig. 5 statistics in `phase_trans_from_image_2_z3/results/`, and every figure, including the Fig. 2 transition panels in `figs/`. They were produced by the commands in the folder READMEs, which also say which cached files have to be deleted to recompute them.

Not included: the raw simulation data, Hugging Face `MLDS-NUS/Wis_ParamSweep_New` and `MLDS-NUS/Wis_Holdout_New` (public), and the image features of simulation frames rendered from them (`data/sim_cache/`). The raw data are needed to regenerate the reduced data and the physical observables, and to redraw Fig. 5, which is drawn from the rendered features. **To reproduce Fig. 1(b), Fig. 4 and Fig. 5 from the raw data, first run `load_dataset("MLDS-NUS/Wis_ParamSweep_New")` once** (about 190 GB download plus about 200 GB Arrow cache):

```bash
export HF_HOME=/path/to/huggingface_cache  # LOCAL: download directory
python -c 'from datasets import load_dataset; load_dataset("MLDS-NUS/Wis_ParamSweep_New")'
```

## Dependencies Between Folders

- `FreeEnergybyIntegration/` figure scripts read outputs of the free-energy block in `examples/UIDD_polymer_dynamics_wi/`; run that block first.
- Fig. 5 in `phase_trans_from_image_2_z3/` reads `z2-z3-Interpret/cache/observables_stride50.npz` from `examples/UIDD_polymer_dynamics_wi/`.
- `phase_trans_from_image_2_z3/` uses the shared code and feature caches of `phase_trans_from_image/`.

## License

MIT, see `LICENSE`.
