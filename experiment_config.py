
"""Frozen MedRIFT experiment configuration used by the manuscript."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("MEDRIFT_DATA_DIR", ROOT / "data")).resolve()
RESOURCE_DIR = ROOT / "resources"
MANIFEST_DIR = RESOURCE_DIR / "manifests"
GRAPH_DIR = RESOURCE_DIR / "graphs"
SPLIT_DIR = ROOT / "splits"
RESULTS_DIR = ROOT / "results"

RANDOM_SEED = 42
N_OUTER_FOLDS = 5
VALIDATION_FRACTION_OF_TRAINVAL = 0.125
SMOKE_TEST = os.environ.get("MEDRIFT_SMOKE_TEST", "0") == "1"
PROTOCOL_ID = "MedRIFT-paper-protocol-v2-20260820"

DATASETS = [
    "cvd",
    "nhanes_diabetes",
    "mi_mortality",
    "support2_mortality",
]

XGB_FIXED_PARAMS = {
    "learning_rate": 0.03,
    "max_depth": 4,
    "min_child_weight": 3.0,
    "subsample": 0.80,
    "colsample_bytree": 0.80,
    "reg_alpha": 0.0,
    "reg_lambda": 1.0,
}
XGB_MAX_ESTIMATORS = 3000
XGB_EARLY_STOPPING_ROUNDS = 100

GPU_IDS = None
PARALLEL_FOLDS = True

PRIMARY_METRICS = ["auprc", "auroc", "macro_f1", "balanced_accuracy"]
CORE_ABLATIONS = [
    "supervised_graph",
    "no_diagnosis_weighting",
    "no_prototype",
    "uniform_gate",
]
GRAPH_ABLATIONS = ["node_only", "rewired", "medical_only", "group_only"]
SENSITIVITY = {
    "beta": [0.0, 0.75, 1.5, 2.25, 3.0],
    "gamma": [0.0, 0.375, 0.75, 1.125, 1.5],
    "lambda": [0.0, 0.1, 0.2, 0.3, 0.4],
}

