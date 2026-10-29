# Neural Network Training and Figures

## Data

Reduced datasets are included in `FinalCODE/data/`:

- `reduced_symm_new_train/`: 111,450 training segments, 500 frames each.
- `reduced_symm_new_test/`: 30 test splits, 500 trajectories each.

## Setup

Requires an NVIDIA GPU. Replace values marked `LOCAL`; skip environment creation if it already exists.

```bash
cd /path/to/FinalCODE  # LOCAL: project directory
conda env create -f environment.gpu-cuda12.yml
conda activate phase-diagram-gpu-cuda12
cd examples/UIDD_polymer_dynamics_wi
export GPU=0  # LOCAL: available GPU index
```

Run all commands below from this example directory. `../../data` refers to the included datasets.

## Generate Reduced Data (Optional)

Use the included reduced data for the original training inputs. Optional regeneration uses the saved PCA parameters and writes to `../../data/regenerated`; PCA coordinates can vary slightly with numerical precision.

Existing raw data in `HF_HOME` are reused. A first-time download needs about 190 GB for `MLDS-NUS/Wis_ParamSweep_New` and 90 GB for `MLDS-NUS/Wis_Holdout_New`, plus about as much again for the Arrow cache that `datasets` builds from them.

`LOCAL`: also set the `HF_HOME` assignment in `generate_reduced_data.py`; it overrides the shell variable below.

```bash
export HF_HOME=/path/to/huggingface_cache  # LOCAL: existing raw-data cache, or download directory
mkdir -p ../../data/regenerated
cp data_cache_nolog/pca_components.pkl ../../data/regenerated/reduced_symm_new_pca_components.pkl &&
CUDA_VISIBLE_DEVICES="$GPU" XLA_PYTHON_CLIENT_PREALLOCATE=false \
  JAX_DEFAULT_MATMUL_PRECISION=highest \
  python -B generate_reduced_data.py \
    data.cache_path=../../data/regenerated \
    data.reduction.load_existing_pca=true \
    hydra.job.chdir=false
```

## Training

Train all five models sequentially; use a single seed to train one model. To use regenerated data, set `data.cache_path=../../data/regenerated`.

```bash
for seed in 123 1 12 1234 12345; do
  CUDA_VISIBLE_DEVICES="$GPU" XLA_PYTHON_CLIENT_PREALLOCATE=false \
    python -B I2_polymer_dynamics_wi.py \
    data.cache_path=../../data \
    model.seed="$seed" model.load_model=false \
    train.seed=123 train.num_epochs=300 \
    hydra.job.chdir=false || break
done
```

Each run saves `model.eqx`, `.hydra/config.yaml`, and logs under `outputs/<timestamp>/`. Point subsequent analysis scripts to these new model directories.

## Figures

Run each block in order from this example directory. The outputs of every block below, intermediate data and figures, are included, so the results can be inspected without running anything. Rerunning a block overwrites them, except for the two cached inputs noted below (surface data and physical observables), which are reused when present.

### Free Energy and Transitions

The saved models under `outputs/` can be used directly. For new training runs, update `MODEL_PATHS` in both `compute_z2_curves.py` and `compute_z3_curves.py` (`LOCAL`; seed order: 123, 1, 12, 1234, 12345).

```bash
export CUDA_VISIBLE_DEVICES="$GPU" XLA_PYTHON_CLIENT_PREALLOCATE=false
export MPLBACKEND=Agg

# Generate curves for all five models; these are the expensive steps.
python -B 1-trans-z2/compute_z2_curves.py
python -B 2-trans-z3/compute_z3_curves.py

python -B 1-trans-z2/plot_z2_argmin.py
python -B 2-trans-z3/plot_z3_argmin.py
python -B plot_transition_combined.py

MODEL_IDX=1 python -B 2-trans-z3/plot_surface_grid.py
MODEL_IDX=1 python -B 2-trans-z3/plot_surface_grid_panels.py
```

Outputs: `transition_combined.{png,pdf}`, `2-trans-z3/F_z2_z3_grid_seed1.{png,pdf,svg}`, and seven separate panels under `figs/`. Surface plots use seed 1 and reuse the included surface data (`1-trans-z2/F_z2_z3_surface_seed1_data.npz`, `2-trans-z3/F_z3_slice_surface_seed1_data.npz`) when their settings match; delete these two files to recompute them.

For separate transition panels including the theoretical comparison, also generate the two theoretical datasets:

```bash
python -B ../../FreeEnergybyIntegration/1-trans-z2/wi_001_10/run_ninf_wi_001_10.py
python -B ../../FreeEnergybyIntegration/2-trans-z3/wi_001_10/run_ninf_joint_z3_wi_001_10.py
python -B ../../FreeEnergybyIntegration/plot_transition_2x2_panels.py
```

These panels are saved as `../../figs/transition_*.{png,pdf,svg}`.

### Polymer Traces and Physical Observables

The physical observables `z2-z3-Interpret/cache/observables_stride50.npz` are included, so the figures can be drawn directly:

```bash
TEST_ROOT="$PWD/../../data/reduced_symm_new_test" \
SWEEP_FULL_DIR="$PWD/../../data/reduced_symm_new_train" \
  python -B plot_polymer/plot_z1_time.py
python -B z2-z3-Interpret/plot_interpret.py wi
```

**To reproduce Fig. 1(b), Fig. 4 and Fig. 5 from the raw data, first run `load_dataset("MLDS-NUS/Wis_ParamSweep_New")` once.** It downloads the parquet files (about 190 GB) and builds the Arrow cache (about 200 GB) that the extractor reads; Fig. 5 also renders from these parquet files (see `../../phase_trans_from_image_2_z3/README.md`). Then recompute the observables and rerun the two plotting commands above:

```bash
export HF_HOME=/path/to/huggingface_cache  # LOCAL: download directory
python -c 'from datasets import load_dataset; load_dataset("MLDS-NUS/Wis_ParamSweep_New")'

rm z2-z3-Interpret/cache/observables_stride50.npz  # the extractor skips an existing file
export RAW_DIR="$HF_HOME/datasets/MLDS-NUS___wis_param_sweep_new/default/0.0.0"  # holds one snapshot directory of Arrow shards
PCA_FILE="$PWD/data_cache_nolog/pca_components.pkl" \
  python -B z2-z3-Interpret/extract_observables.py
```

The extractor creates `z2-z3-Interpret/cache/observables_stride50.npz`. Figures are saved beside the plotting scripts, including `z1_traces_single.png` and `z2_z3_vs_physical_by_wi.{png,pdf,svg}`.

### Prediction Statistics

Open the analysis notebook using the current conda environment:

```bash
python -m notebook polymer_dynamics_wi_symm_analysis.ipynb
```

`LOCAL`: set its GPU index, `HF_HOME`, model `path`, and `test_data_name="../../data/reduced_symm_new_test"`. Run through the cell that saves `symm_analysis_stats_*.npz`. Set `plot_file` in `plot_test/plot_test.ipynb` to `../symm_analysis_stats_<model-directory>.npz`, then plot:

```bash
(cd plot_test && python -B -m nbconvert --to notebook --execute --stdout \
  --ExecutePreprocessor.kernel_name=python3 \
  --ExecutePreprocessor.timeout=-1 plot_test.ipynb > /dev/null)
```

Output: `plot_test/symm_analysis_stats_<model-directory>.pdf`.
