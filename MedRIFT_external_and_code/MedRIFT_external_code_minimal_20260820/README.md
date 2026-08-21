# 此文件更新于 2026-08-20 15:26（北京时间）

# MedRIFT

This repository contains the implementation required to train and evaluate
MedRIFT as described in the paper. It includes the fixed model configuration,
fold-local preprocessing, cross-fitted tree diagnosis, tree-region construction,
medical-graph interaction encoder, residual-weighted classification,
leave-one-out region-class prototype loss, final tree fitting, and the reported
ablation protocols.

No patient-level or public-dataset records are distributed in this repository.

## Files

- `medrift_engine.py`: complete MedRIFT training and evaluation pipeline.
- `experiment_config.py`: fixed protocol and hyperparameters.
- `experiment_core.py`: patient-disjoint folds, metrics, and result persistence.
- `run_medrift.py`: runs the main MedRIFT experiment.
- `run_ablations.py`: runs the core, graph, and parameter-sensitivity studies.
- `preprocess_public_datasets.py`: constructs the three public analysis tables
  from locally downloaded source files.
- `resources/manifests/`: targets, feature lists, exclusions, and preprocessing
  rules used by the four tasks.
- `resources/graphs/`: variable-to-LOINC mappings, selected LOINC panels,
  clinical groups, and the fixed graph edge sets.
- `BASELINES.md`: versions and execution policy for the compared methods.

## Environment

The reported experiments used Python 3.9, PyTorch 2.6, CUDA 12.6,
scikit-learn 1.6.1, and XGBoost 2.1.4. Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

Use the PyTorch wheel appropriate for the local CUDA installation when the
default package index does not provide a compatible build.

## Data placement

By default, processed input tables are read from `data/`. An alternative
directory can be supplied through the `MEDRIFT_DATA_DIR` environment variable.
The required filenames are:

```text
cvd.csv
nhanes_diabetes.csv
mi_mortality.csv
support2_mortality.csv
```

The private CVD cohort is not distributed. Its required columns are defined in
`resources/manifests/cvd_manifest.json`. The public-data preprocessing script
expects the following locally downloaded files:

```text
source_raw/nhanes/{DEMO_J,DIQ_J,BMX_J,CBC_J,BIOPRO_J,HDL_J,GHB_J}.xpt
source_raw/mi/MI.data
source_raw/support2/support2_raw.csv
```

Run:

```bash
python preprocess_public_datasets.py
```

The script performs deterministic cohort construction and feature selection.
Learned imputation, numerical standardization, categorical encoding, truncated
SVD, and k-means are fitted separately inside each outer training fold by the
experiment pipeline.

## Main experiment

Run all configured datasets:

```bash
python run_medrift.py
```

To run one dataset, set `MEDRIFT_DATASET` to `cvd`, `nhanes_diabetes`,
`mi_mortality`, or `support2_mortality`, then run `medrift_engine.py`.
The program uses five stratified outer folds and assigns independent outer folds
to the visible GPUs. Completed folds are reused only when their protocol ID
matches the current configuration.

## Ablations

```bash
python run_ablations.py
```

This command covers the supervised graph encoder, residual-weight removal,
prototype removal, uniform edge gating, node-only graph, degree-preserving
rewired graph, single-source graphs, and sensitivity to `beta`, `gamma`, and
`lambda`. Outputs are written under `results/`; generated splits are stored
under `splits/`.

## Medical graph

Graph construction is fixed to LOINC 2.82. `E_med` contains edges induced by
selected LOINC panel co-membership, `E_group` contains edges induced by fixed
clinical functional groups, and the model uses their deduplicated union. No
training-set correlation edges are used. The repository contains only the
task-specific mappings and graph definitions, not a copy of the LOINC release.

