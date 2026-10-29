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
| `examples/UIDD_polymer_dynamics_wi/` | Neural-network pipeline for the polymer data. |
| `FreeEnergybyIntegration/` | Analytical model: long-chain limit sweeps over Wi, finite-n Monte Carlo at the figure points. |
| `phase_trans_from_image/` | Shared image code. |
| `phase_trans_from_image_2_z3/` | Experimental analysis. |
| `Langevin_single_chain/` | `microscPSF.py`: Gibson–Lanni point-spread function, used to render simulated frames. | 
| `data/` | Input data and feature caches, see below. |

## Data

Included in `data/`:

- `reduced_symm_new_train/`, `reduced_symm_new_test/`: reduced simulation datasets (Hugging Face `datasets` format) for training and testing.
- `exp_data/cache_voltage_frames/`: per-frame image features of the experimental videos.

The raw experimental frames are the public Hugging Face dataset [`MLDS-NUS/polymer-exp-voltage-frames`](https://huggingface.co/datasets/MLDS-NUS/polymer-exp-voltage-frames) (about 30 GB): `10V/` (48 video tars of PNG frames) and `20V_2_Feb25.pkl`, `40V_Feb25.pkl`, `50V_Feb25.pkl`. The scripts read them from `data/exp_data/voltage-frames/`; download them there with the `hf` command of either environment:

```bash
hf download MLDS-NUS/polymer-exp-voltage-frames --repo-type dataset --local-dir data/exp_data/voltage-frames
```


Not included: the raw simulation data, Hugging Face `MLDS-NUS/Wis_ParamSweep_New` and `MLDS-NUS/Wis_Holdout_New` (public). The raw data are needed to regenerate the reduced data and the physical observables 

```bash
export HF_HOME=/path/to/huggingface_cache  # LOCAL: download directory
python -c 'from datasets import load_dataset; load_dataset("MLDS-NUS/Wis_ParamSweep_New")'
```

## License

MIT, see `LICENSE`.
