
"""Run the locked MedRIFT + XGBoost protocol on four medical datasets.

Keep this file beside the ``data`` folder and click Run in PyCharm. No command
line arguments are required. The implementation follows the locked method:

1. cross-fitted (OOF) XGBoost residual diagnosis;
2. reference-tree leaf indicators -> truncated SVD -> K-means regions;
3. normalized region ambiguity;
4. fixed medical graph E = E_med union E_group (never correlation edges);
5. patient-specific residual interaction encoder;
6. residual-weighted classification + stop-gradient region-class prototypes;
7. independent final XGBoost on [original tree features; residual features].

The outer evaluation splits are identical to ``run_all_xgboost.py``.
"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import joblib
    import numpy as np
    import pandas as pd
    import sklearn
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import xgboost
    from sklearn.cluster import KMeans
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import TruncatedSVD
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        brier_score_loss,
        confusion_matrix,
        f1_score,
        roc_auc_score,
    )
    from sklearn.model_selection import StratifiedKFold, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from torch.utils.data import DataLoader, TensorDataset
    from xgboost import XGBClassifier
except ImportError as exc:
    raise SystemExit(
        "缺少运行库。请在PyCharm当前环境的Terminal执行：\n"
        "pip install numpy pandas scikit-learn xgboost torch joblib\n\n"
        f"原始错误：{exc}"
    ) from exc

from experiment_core import (build_or_validate_splits, fold_is_complete, load_dataset,
                             negative_average_precision, youden_threshold)
from experiment_config import (DATASETS, DATA_DIR, GRAPH_DIR, MANIFEST_DIR,
                               PRIMARY_METRICS, PROTOCOL_ID, RESULTS_DIR, SMOKE_TEST,
                               XGB_EARLY_STOPPING_ROUNDS, XGB_FIXED_PARAMS,
                               XGB_MAX_ESTIMATORS)


# ======================== Direct-run configuration ========================
BASE_DIR = Path(__file__).resolve().parent
EXPERIMENT_VARIANT = os.environ.get("MEDRIFT_VARIANT", "full")
GRAPH_VARIANT = os.environ.get(
    "MEDRIFT_GRAPH", "node_only" if EXPERIMENT_VARIANT == "node_only" else "full"
)
OUTPUT_ROOT = RESULTS_DIR / ("medrift_smoke" if SMOKE_TEST else "medrift") / EXPERIMENT_VARIANT
DATASETS_TO_RUN = list(DATASETS)

# These globals are filled from each dataset manifest before any training.
DATASET_STEM = ""
DATA_PATH = Path()
MANIFEST_PATH = Path()
GRAPH_PATH = Path()
OUTPUT_DIR = Path()
DATASET_NAME = ""
TASK_NAME = ""
RANDOM_SEED = 42
N_OUTER_FOLDS = 5
VALIDATION_FRACTION_OF_TRAINVAL = 0.125
N_DIAGNOSTIC_FOLDS = 5
FINAL_XGB_MAX_ESTIMATORS = 30 if SMOKE_TEST else XGB_MAX_ESTIMATORS
FINAL_XGB_EARLY_STOPPING_ROUNDS = 5 if SMOKE_TEST else XGB_EARLY_STOPPING_ROUNDS

# Windows multi-GPU execution. Each GPU receives one independent worker and
# one or more complete outer folds. Set False for strictly serial execution.
USE_ALL_GPUS = True
# None means all GPUs visible to this Python process. Example: [0, 2].
GPU_IDS = None
# XGBoost >= 2.0 can use the same GPU assigned to the fold worker. Older
# versions automatically keep XGBoost on CPU while PyTorch still uses GPU.
USE_GPU_FOR_XGBOOST = True
# 0 divides logical CPU cores across active GPU workers automatically.
CPU_THREADS_PER_WORKER = 0

FEATURES: list[str] = []
CATEGORICAL_FEATURES: list[str] = []
NUMERICAL_FEATURES: list[str] = []
MISSING_FEATURES: list[str] = []

DIAGNOSTIC_METRICS = ["sensitivity", "specificity", "threshold", "brier"]


# Shared fixed tree settings. There is no dataset-specific tree HPO.
XGB_PARAMS = {
    "n_estimators": 500,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": RANDOM_SEED,
    **XGB_FIXED_PARAMS,
}

XGB_FIXED_CANDIDATE = {"name": "fixed_shared", **XGB_FIXED_PARAMS}


@dataclass(frozen=True)
class EncoderConfig:
    node_dim: int = 16
    edge_dim: int = 32
    residual_dim: int = 12
    dropout: float = 0.15
    batch_size: int = 256
    epochs: int = 300  # maximum; validation early stopping normally ends earlier
    minimum_epochs: int = 30
    early_stopping_patience: int = 25
    early_stopping_min_delta: float = 1e-4
    learning_rate: float = 7e-4
    warmup_epochs: int = 5
    lr_decay_epochs: int = 50
    minimum_lr_ratio: float = 0.05
    weight_decay: float = 2e-4
    beta: float = 1.5
    gamma: float = 0.75
    prototype_weight: float = 0.20
    temperature: float = 0.20
    minimum_prototype_size: int = 4
    path_embedding_dim: int = 16
    tree_regions: int = 8
    uniform_gating: bool = False
    node_only: bool = False


ENCODER_CONFIG = EncoderConfig()


def encoder_config_for_dataset(stem: str) -> tuple[str, EncoderConfig]:
    """Return one frozen encoder protocol; validation selects epochs only."""
    if stem not in DATASETS_TO_RUN:
        raise ValueError(f"未知数据集：{stem}")
    cfg = ENCODER_CONFIG
    if stem == "cvd":
        # The private CVD cohort is substantially larger than the public
        # datasets. This keeps the same architecture and objectives while
        # using a larger batch and validation-AUPRC early stopping.
        cfg = replace(
            cfg,
            batch_size=1024,
            epochs=60,
            minimum_epochs=5,
            early_stopping_patience=5,
            warmup_epochs=3,
            lr_decay_epochs=20,
        )
    if EXPERIMENT_VARIANT == "supervised_graph":
        cfg = replace(cfg, beta=0.0, gamma=0.0, prototype_weight=0.0)
    elif EXPERIMENT_VARIANT == "no_diagnosis_weighting": cfg = replace(cfg, beta=0.0, gamma=0.0)
    elif EXPERIMENT_VARIANT == "no_prototype": cfg = replace(cfg, prototype_weight=0.0)
    elif EXPERIMENT_VARIANT == "uniform_gate": cfg = replace(cfg, uniform_gating=True)
    if GRAPH_VARIANT == "node_only": cfg = replace(cfg, node_only=True)
    if os.environ.get("MEDRIFT_K"): cfg = replace(cfg, tree_regions=int(os.environ["MEDRIFT_K"]))
    if os.environ.get("MEDRIFT_DZ"): cfg = replace(cfg, residual_dim=int(os.environ["MEDRIFT_DZ"]))
    if os.environ.get("MEDRIFT_BETA"): cfg = replace(cfg, beta=float(os.environ["MEDRIFT_BETA"]))
    if os.environ.get("MEDRIFT_GAMMA"): cfg = replace(cfg, gamma=float(os.environ["MEDRIFT_GAMMA"]))
    if os.environ.get("MEDRIFT_LAMBDA"): cfg = replace(cfg, prototype_weight=float(os.environ["MEDRIFT_LAMBDA"]))
    if SMOKE_TEST:
        cfg = replace(cfg, epochs=3, minimum_epochs=2, early_stopping_patience=1,
                      warmup_epochs=1, lr_decay_epochs=2, minimum_prototype_size=2)
    return "fixed_shared", cfg


def configure_dataset(stem: str) -> None:
    """Load one manifest and expose its schema to the shared implementation."""
    global DATASET_STEM, DATA_PATH, MANIFEST_PATH, GRAPH_PATH, OUTPUT_DIR
    global DATASET_NAME, TASK_NAME
    global FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES, MISSING_FEATURES

    manifest_path = MANIFEST_DIR / f"{stem}_manifest.json"
    graph_path = GRAPH_DIR / f"{stem}_graph.json"
    data_path = DATA_DIR / f"{stem}.csv"
    for path in (manifest_path, graph_path, data_path):
        if not path.exists():
            raise FileNotFoundError(f"缺少必要文件：{path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    continuous = list(manifest["continuous_features"])
    categorical = list(manifest["categorical_features"])
    features = continuous + categorical
    if len(features) != len(set(features)):
        raise ValueError(f"{stem} manifest中存在重复特征。")

    DATASET_STEM, DATA_PATH = stem, data_path
    MANIFEST_PATH, GRAPH_PATH = manifest_path, graph_path
    OUTPUT_DIR = OUTPUT_ROOT / stem
    DATASET_NAME, TASK_NAME = manifest["dataset"], manifest["task"]
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES = continuous, categorical
    FEATURES = features
    MISSING_FEATURES = [f"{name}__missing" for name in features]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def locate_data() -> Path:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"找不到处理后的CSV：{DATA_PATH}")
    return DATA_PATH


def read_data(path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    data, labels, patient_ids, _ = load_dataset(DATASET_STEM, path)
    counts = np.bincount(labels, minlength=2)
    if counts.min() < max(N_OUTER_FOLDS, N_DIAGNOSTIC_FOLDS):
        raise ValueError(
            "阳性或阴性样本太少，无法进行当前分层交叉验证："
            f"阴性={counts[0]}，阳性={counts[1]}。"
        )

    return data, labels, patient_ids


def make_tree_preprocessor() -> ColumnTransformer:
    try:
        onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn < 1.2
        onehot = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERICAL_FEATURES),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", onehot),
                ]),
                CATEGORICAL_FEATURES,
            ),
            ("missing", "passthrough", MISSING_FEATURES),
        ],
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def positive_class_weight(y: np.ndarray) -> float:
    positive = int((y == 1).sum())
    if positive == 0:
        raise ValueError("训练集中没有阳性样本。")
    return int((y == 0).sum()) / positive


def xgboost_supports_device() -> bool:
    try:
        return int(xgboost.__version__.split(".")[0]) >= 2
    except (TypeError, ValueError):
        return False


def make_xgboost(
    y_train: np.ndarray,
    seed: int,
    early_stopping: bool = False,
    candidate: Optional[dict] = None,
    use_class_weight: bool = True,
) -> XGBClassifier:
    params = dict(XGB_PARAMS)
    if candidate is not None:
        params.update({key: value for key, value in candidate.items() if key != "name"})
    params["random_state"] = seed
    params["scale_pos_weight"] = positive_class_weight(y_train) if use_class_weight else 1.0
    if early_stopping:
        params["n_estimators"] = FINAL_XGB_MAX_ESTIMATORS
        params["eval_metric"] = negative_average_precision
        params["early_stopping_rounds"] = FINAL_XGB_EARLY_STOPPING_ROUNDS
    worker_threads = int(os.environ.get("MEDRIFT_CPU_THREADS", "0"))
    if worker_threads > 0:
        params["n_jobs"] = worker_threads
    if USE_GPU_FOR_XGBOOST and torch.cuda.is_available() and xgboost_supports_device():
        # A multi-GPU worker sees only its assigned GPU, remapped to cuda:0.
        params["device"] = "cuda"
    return XGBClassifier(**params)


def fit_fixed_xgboost(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    seed: int,
    use_class_weight: bool,
) -> tuple[XGBClassifier, dict, list[dict]]:
    model = make_xgboost(
        y_train, seed, early_stopping=True, candidate=XGB_FIXED_CANDIDATE,
        use_class_weight=use_class_weight,
    )
    model.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)
    probability = model.predict_proba(x_valid)[:, 1]
    record = {
        "candidate": XGB_FIXED_CANDIDATE["name"],
        "validation_auprc": float(average_precision_score(y_valid, probability)),
        "best_iteration": int(getattr(model, "best_iteration", FINAL_XGB_MAX_ESTIMATORS - 1)),
        "scale_pos_weight": positive_class_weight(y_train) if use_class_weight else 1.0,
        "max_estimators": FINAL_XGB_MAX_ESTIMATORS,
        "early_stopping_rounds": FINAL_XGB_EARLY_STOPPING_ROUNDS,
        **XGB_FIXED_PARAMS,
    }
    return model, record, [record]


class NeuralPreprocessor:
    """Train-only imputation/scaling while preserving one node per variable."""

    def __init__(self) -> None:
        self.medians: dict[str, float] = {}
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}
        self.categories: dict[str, list[str]] = {}

    def fit(self, frame: pd.DataFrame) -> "NeuralPreprocessor":
        for feature in NUMERICAL_FEATURES:
            values = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            median = float(np.median(finite)) if finite.size else 0.0
            filled = np.where(np.isfinite(values), values, median)
            mean = float(filled.mean())
            std = float(filled.std())
            self.medians[feature] = median
            self.means[feature] = mean
            self.stds[feature] = std if std > 1e-8 else 1.0
        for feature in CATEGORICAL_FEATURES:
            values = frame[feature].dropna().astype(str)
            self.categories[feature] = sorted(values.unique().tolist())
        return self

    def transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        n, d = len(frame), len(FEATURES)
        values = np.zeros((n, d), dtype=np.float32)
        masks = np.zeros((n, d), dtype=np.float32)
        for j, feature in enumerate(FEATURES):
            mask_name = f"{feature}__missing"
            masks[:, j] = pd.to_numeric(frame[mask_name], errors="coerce").fillna(1).to_numpy(np.float32)
            if feature in NUMERICAL_FEATURES:
                raw = pd.to_numeric(frame[feature], errors="coerce").to_numpy(dtype=float)
                raw = np.where(np.isfinite(raw), raw, self.medians[feature])
                values[:, j] = ((raw - self.means[feature]) / self.stds[feature]).astype(np.float32)
            else:
                mapping = {category: code + 1 for code, category in enumerate(self.categories[feature])}
                raw = frame[feature].astype("string")
                values[:, j] = raw.map(mapping).fillna(0).to_numpy(np.float32)
        return values, masks

    def category_cardinalities(self) -> dict[int, int]:
        return {
            FEATURES.index(feature): len(self.categories[feature]) + 1
            for feature in CATEGORICAL_FEATURES
        }


def degree_preserving_rewire(
    edges: set[tuple[int, int]], seed: int, accepted_swaps: int
) -> set[tuple[int, int]]:
    """Rewire an undirected simple graph while preserving every node degree."""
    rng = random.Random(seed)
    current = set(edges)
    edge_list = sorted(current)
    accepted = 0
    attempts = 0
    maximum_attempts = max(1000, accepted_swaps * 200)
    while accepted < accepted_swaps and attempts < maximum_attempts:
        attempts += 1
        first, second = rng.sample(range(len(edge_list)), 2)
        a, b = edge_list[first]
        c, d = edge_list[second]
        if len({a, b, c, d}) != 4:
            continue
        if rng.random() < 0.5:
            proposed = {(min(a, d), max(a, d)), (min(c, b), max(c, b))}
        else:
            proposed = {(min(a, c), max(a, c)), (min(b, d), max(b, d))}
        old = {(a, b), (c, d)}
        if len(proposed) != 2 or proposed & (current - old):
            continue
        current.difference_update(old)
        current.update(proposed)
        edge_list = sorted(current)
        accepted += 1
    if accepted != accepted_swaps:
        raise RuntimeError(
            f"Degree-preserving rewiring accepted {accepted}/{accepted_swaps} swaps."
        )
    return current


def build_graph(fold: Optional[int] = None) -> tuple[np.ndarray, dict[str, object]]:
    provenance = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    if set(provenance.get("nodes", [])) != set(FEATURES):
        missing = sorted(set(FEATURES) - set(provenance.get("nodes", [])))
        extra = sorted(set(provenance.get("nodes", [])) - set(FEATURES))
        raise ValueError(f"知识图节点与manifest不一致。缺失={missing}，多余={extra}")
    index = {feature: j for j, feature in enumerate(FEATURES)}
    graph_variant = GRAPH_VARIANT
    if graph_variant == "node_only":
        provenance.update(active_graph_variant=graph_variant, active_edges=[])
        return np.empty((0, 2), dtype=np.int64), provenance
    graph_key = {"full": "E", "medical_only": "E_med", "group_only": "E_group",
                 "rewired": "E"}.get(graph_variant)
    if graph_key is None:
        raise ValueError(f"未知图版本：{graph_variant}")
    edge_names = provenance.get(graph_key, [])
    if not edge_names:
        raise ValueError(f"{GRAPH_PATH.name} 中没有知识图边。")
    edges: set[tuple[int, int]] = set()
    for pair in edge_names:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"非法知识图边：{pair}")
        left, right = pair
        if left not in index or right not in index or left == right:
            raise ValueError(f"知识图边包含未知节点或自环：{pair}")
        a, b = index[left], index[right]
        edges.add((min(a, b), max(a, b)))
    if graph_variant == "rewired":
        if fold is None:
            raise ValueError("The rewired graph requires an outer-fold index.")
        target_swaps = 10 * len(edges)
        edges = degree_preserving_rewire(edges, RANDOM_SEED + fold, target_swaps)
    provenance = dict(provenance)
    provenance["active_graph_variant"] = graph_variant
    if graph_variant == "rewired":
        provenance["rewiring_seed"] = RANDOM_SEED + fold
        provenance["accepted_double_edge_swaps"] = 10 * len(edges)
    provenance["active_edges"] = [list(e) for e in sorted(edges)]
    return np.asarray(sorted(edges), dtype=np.int64), provenance


class VariableEncoder(nn.Module):
    def __init__(self, hidden_dim: int, category_count: Optional[int]) -> None:
        super().__init__()
        self.category_count = category_count
        if category_count is None:
            self.numeric = nn.Linear(2, hidden_dim)
            self.embedding = None
            self.missing_projection = None
        else:
            self.numeric = None
            self.embedding = nn.Embedding(category_count, hidden_dim)
            self.missing_projection = nn.Linear(1, hidden_dim, bias=False)
        self.normalization = nn.LayerNorm(hidden_dim)

    def forward(self, value: torch.Tensor, missing: torch.Tensor) -> torch.Tensor:
        if self.embedding is None:
            hidden = self.numeric(torch.stack([value, missing], dim=1))
        else:
            code = value.long().clamp(min=0, max=self.category_count - 1)
            hidden = self.embedding(code) + self.missing_projection(missing.unsqueeze(1))
        return self.normalization(F.gelu(hidden))


class ResidualInteractionEncoder(nn.Module):
    def __init__(
        self,
        category_cardinalities: dict[int, int],
        edge_index: np.ndarray,
        config: EncoderConfig,
    ) -> None:
        super().__init__()
        self.config = config
        self.nodes = nn.ModuleList([
            VariableEncoder(config.node_dim, category_cardinalities.get(j))
            for j in range(len(FEATURES))
        ])
        self.edge_encoder = nn.Sequential(
            nn.Linear(4 * config.node_dim, config.edge_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.edge_dim, config.edge_dim),
            nn.GELU(),
        )
        self.edge_gate = nn.Linear(config.edge_dim, 1)
        self.node_pool_projection = nn.Linear(config.node_dim, config.edge_dim)
        self.projection = nn.Sequential(
            nn.Linear(config.edge_dim, config.residual_dim),
            nn.GELU(),
            nn.LayerNorm(config.residual_dim),
        )
        self.classifier = nn.Linear(config.residual_dim, 2)
        self.register_buffer("edge_left", torch.as_tensor(edge_index[:, 0], dtype=torch.long))
        self.register_buffer("edge_right", torch.as_tensor(edge_index[:, 1], dtype=torch.long))

    def encode(self, values: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        nodes = torch.stack(
            [module(values[:, j], masks[:, j]) for j, module in enumerate(self.nodes)],
            dim=1,
        )
        if self.config.node_only:
            return self.projection(F.gelu(self.node_pool_projection(nodes.mean(dim=1))))
        left = nodes[:, self.edge_left, :]
        right = nodes[:, self.edge_right, :]
        descriptor = torch.cat([left, right, left * right, torch.abs(left - right)], dim=2)
        edge_hidden = self.edge_encoder(descriptor)
        gates = (torch.ones_like(edge_hidden[:, :, :1]) if self.config.uniform_gating
                 else torch.sigmoid(self.edge_gate(edge_hidden)))
        pooled = (gates * edge_hidden).sum(dim=1) / (gates.sum(dim=1) + 1e-8)
        return self.projection(pooled)

    def forward(self, values: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        representation = self.encode(values, masks)
        return representation, self.classifier(representation)


def diagnostic_oof_probabilities(
    frame: pd.DataFrame,
    labels: np.ndarray,
    valid_frame: pd.DataFrame,
    valid_labels: np.ndarray,
    seed: int,
) -> np.ndarray:
    class_counts = np.bincount(labels, minlength=2)
    n_splits = int(min(N_DIAGNOSTIC_FOLDS, class_counts.min()))
    if n_splits < 2:
        raise ValueError("诊断训练集的某一类别少于2例，无法生成OOF预测。")
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    probabilities = np.full(len(labels), np.nan, dtype=np.float64)
    for inner_fold, (fit_idx, held_idx) in enumerate(splitter.split(frame, labels), start=1):
        preprocessor = make_tree_preprocessor()
        fit_x = preprocessor.fit_transform(frame.iloc[fit_idx]).astype(np.float32)
        held_x = preprocessor.transform(frame.iloc[held_idx]).astype(np.float32)
        valid_x = preprocessor.transform(valid_frame).astype(np.float32)
        # No class weighting here: residual scores require probabilities whose
        # scale is not intentionally shifted by imbalance reweighting.
        model = make_xgboost(
            labels[fit_idx], seed + inner_fold,
            early_stopping=True,
            candidate=XGB_FIXED_CANDIDATE,
            use_class_weight=False,
        )
        model.fit(fit_x, labels[fit_idx], eval_set=[(valid_x, valid_labels)], verbose=False)
        probabilities[held_idx] = model.predict_proba(held_x)[:, 1]
    if not np.isfinite(probabilities).all():
        raise RuntimeError("OOF诊断概率没有覆盖全部训练患者。")
    return probabilities


def build_tree_regions(
    reference_model: XGBClassifier,
    tree_train: np.ndarray,
    labels: np.ndarray,
    config: EncoderConfig,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, OneHotEncoder, TruncatedSVD, KMeans]:
    leaves = np.asarray(reference_model.apply(tree_train))
    if leaves.ndim == 1:
        leaves = leaves.reshape(-1, 1)
    try:
        leaf_encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=True, dtype=np.float32
        )
    except TypeError:  # scikit-learn < 1.2
        leaf_encoder = OneHotEncoder(
            handle_unknown="ignore", sparse=True, dtype=np.float32
        )
    indicators = leaf_encoder.fit_transform(leaves)
    max_components = min(indicators.shape[0] - 1, indicators.shape[1] - 1)
    if max_components < 1:
        raise ValueError("树叶指示矩阵太小，无法执行Truncated SVD。")
    n_components = min(config.path_embedding_dim, max_components)
    svd = TruncatedSVD(n_components=n_components, random_state=seed)
    embedding = svd.fit_transform(indicators)

    n_regions = min(config.tree_regions, len(labels))
    n_regions = max(2, n_regions)
    kmeans = KMeans(n_clusters=n_regions, n_init=20, random_state=seed)
    regions = kmeans.fit_predict(embedding).astype(np.int64)

    ambiguity = np.zeros(n_regions, dtype=np.float32)
    for region in range(n_regions):
        region_labels = labels[regions == region]
        if region_labels.size == 0:
            continue
        proportions = np.bincount(region_labels, minlength=2).astype(float)
        proportions /= proportions.sum()
        nonzero = proportions[proportions > 0]
        ambiguity[region] = float(-(nonzero * np.log(nonzero)).sum() / math.log(2.0))
    return regions, ambiguity, leaf_encoder, svd, kmeans


@dataclass(frozen=True)
class PrototypeCache:
    sums: torch.Tensor
    counts: torch.Tensor
    normalized_embeddings: torch.Tensor
    valid_region_count: int


@torch.no_grad()
def compute_prototype_cache(
    model: ResidualInteractionEncoder,
    values: torch.Tensor,
    masks: torch.Tensor,
    labels: torch.Tensor,
    regions: torch.Tensor,
    minimum_size: int,
    batch_size: int,
) -> PrototypeCache:
    model.eval()
    chunks = []
    for start in range(0, len(values), batch_size):
        chunks.append(model.encode(values[start:start + batch_size], masks[start:start + batch_size]))
    normalized = F.normalize(torch.cat(chunks, dim=0), p=2, dim=1, eps=1e-8)
    n_regions = int(regions.max().item()) + 1
    sums = torch.zeros(
        (n_regions, 2, normalized.shape[1]), dtype=normalized.dtype,
        device=normalized.device,
    )
    counts = torch.zeros((n_regions, 2), dtype=torch.long, device=normalized.device)
    for region in range(n_regions):
        for class_id in (0, 1):
            selected = (regions == region) & (labels == class_id)
            counts[region, class_id] = selected.sum()
            if selected.any():
                sums[region, class_id] = normalized[selected].sum(dim=0)
    eligible_class_zero = (counts[:, 0] >= minimum_size + 1) & (counts[:, 1] >= minimum_size)
    eligible_class_one = (counts[:, 1] >= minimum_size + 1) & (counts[:, 0] >= minimum_size)
    valid_region_count = int((eligible_class_zero | eligible_class_one).sum().item())
    return PrototypeCache(
        sums=sums.detach(),
        counts=counts.detach(),
        normalized_embeddings=normalized.detach(),
        valid_region_count=valid_region_count,
    )


def prototype_loss(
    representation: torch.Tensor,
    labels: torch.Tensor,
    regions: torch.Tensor,
    sample_indices: torch.Tensor,
    cache: PrototypeCache,
    temperature: float,
    minimum_size: int,
) -> torch.Tensor:
    normalized = F.normalize(representation, p=2, dim=1, eps=1e-8)
    sums = cache.sums[regions].clone()
    counts = cache.counts[regions].clone()
    rows = torch.arange(len(labels), device=labels.device)
    sums[rows, labels] -= cache.normalized_embeddings[sample_indices]
    counts[rows, labels] -= 1
    valid_classes = counts >= minimum_size
    eligible = valid_classes[rows, labels] & (valid_classes.sum(dim=1) >= 2)
    if not eligible.any():
        return representation.sum() * 0.0
    anchors = F.normalize(sums, p=2, dim=2, eps=1e-8)
    logits = torch.einsum("bd,bcd->bc", normalized, anchors) / temperature
    logits = logits.masked_fill(~valid_classes, torch.finfo(logits.dtype).min)
    return F.cross_entropy(logits[eligible], labels[eligible])


def normalize_diagnostic_weights(
    raw_weights: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, dict[int, float]]:
    """Normalize diagnosis-guided weights within each training class."""
    normalized = np.asarray(raw_weights, dtype=np.float32).copy()
    class_means: dict[int, float] = {}
    for class_id in (0, 1):
        selected = labels == class_id
        if not selected.any():
            raise ValueError(f"Training data contain no patients in class {class_id}.")
        class_mean = float(normalized[selected].mean())
        if not np.isfinite(class_mean) or class_mean <= 0:
            raise ValueError(f"Residual weights have an invalid class-{class_id} mean.")
        class_means[class_id] = class_mean
        normalized[selected] /= class_mean
    return normalized, class_means


def train_encoder(
    train_values: np.ndarray,
    train_masks: np.ndarray,
    valid_values: np.ndarray,
    valid_masks: np.ndarray,
    labels: np.ndarray,
    valid_labels: np.ndarray,
    residuals: np.ndarray,
    regions: np.ndarray,
    region_ambiguity: np.ndarray,
    category_cardinalities: dict[int, int],
    edge_index: np.ndarray,
    config: EncoderConfig,
    device: torch.device,
    seed: int,
) -> tuple[ResidualInteractionEncoder, list[dict[str, float]]]:
    set_seed(seed)
    values = torch.as_tensor(train_values, dtype=torch.float32, device=device)
    masks = torch.as_tensor(train_masks, dtype=torch.float32, device=device)
    y = torch.as_tensor(labels, dtype=torch.long, device=device)
    region_tensor = torch.as_tensor(regions, dtype=torch.long, device=device)
    eta_np = 1.0 + config.beta * residuals + config.gamma * region_ambiguity[regions]
    eta_np, eta_class_means = normalize_diagnostic_weights(eta_np, labels)
    eta = torch.as_tensor(eta_np, dtype=torch.float32, device=device)
    validation_values = torch.as_tensor(valid_values, dtype=torch.float32, device=device)
    validation_masks = torch.as_tensor(valid_masks, dtype=torch.float32, device=device)

    model = ResidualInteractionEncoder(category_cardinalities, edge_index, config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    sample_indices = torch.arange(len(values), dtype=torch.long, device=device)
    dataset = TensorDataset(values, masks, y, eta, region_tensor, sample_indices)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=min(config.batch_size, len(dataset)),
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        generator=generator,
    )
    history: list[dict[str, float]] = []
    best_score = -float("inf")
    best_epoch = 0
    best_state: Optional[dict[str, torch.Tensor]] = None
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        if epoch <= config.warmup_epochs:
            lr_factor = epoch / max(config.warmup_epochs, 1)
        else:
            decay_end = max(config.warmup_epochs + 1, min(config.lr_decay_epochs, config.epochs))
            progress = min(
                1.0,
                (epoch - config.warmup_epochs) / (decay_end - config.warmup_epochs),
            )
            lr_factor = config.minimum_lr_ratio + 0.5 * (
                1.0 - config.minimum_lr_ratio
            ) * (1.0 + math.cos(math.pi * progress))
        current_lr = config.learning_rate * lr_factor
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = current_lr

        prototype_cache = compute_prototype_cache(
            model, values, masks, y, region_tensor,
            config.minimum_prototype_size, config.batch_size,
        )
        model.train()
        running_total = running_residual = running_prototype = 0.0
        seen = 0
        for (batch_values, batch_masks, batch_y, batch_eta, batch_regions,
             batch_indices) in loader:
            optimizer.zero_grad(set_to_none=True)
            representation, logits = model(batch_values, batch_masks)
            per_sample = F.cross_entropy(logits, batch_y, reduction="none")
            residual_loss = (batch_eta * per_sample).mean()
            proto_loss = prototype_loss(
                representation, batch_y, batch_regions,
                batch_indices, prototype_cache, config.temperature,
                config.minimum_prototype_size,
            )
            loss = residual_loss + config.prototype_weight * proto_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            batch_n = len(batch_y)
            seen += batch_n
            running_total += float(loss.detach()) * batch_n
            running_residual += float(residual_loss.detach()) * batch_n
            running_prototype += float(proto_loss.detach()) * batch_n
        model.eval()
        validation_probability = []
        with torch.no_grad():
            for start in range(0, len(validation_values), config.batch_size):
                _, validation_logits = model(
                    validation_values[start:start + config.batch_size],
                    validation_masks[start:start + config.batch_size],
                )
                validation_probability.append(
                    torch.softmax(validation_logits, dim=1)[:, 1].cpu().numpy()
                )
        validation_auprc = float(average_precision_score(
            valid_labels, np.concatenate(validation_probability)
        ))
        previous_best = best_score
        if validation_auprc > best_score:
            best_score = validation_auprc
            best_epoch = epoch
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }
        if epoch <= config.minimum_epochs:
            # Complete warmup/cosine exploration before patience is allowed to
            # accumulate; best checkpoints are still recorded during this phase.
            epochs_without_improvement = 0
        elif validation_auprc > previous_best + config.early_stopping_min_delta:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        row = {
            "epoch": epoch,
            "loss": running_total / seen,
            "residual_loss": running_residual / seen,
            "prototype_loss": running_prototype / seen,
            "valid_prototype_regions": prototype_cache.valid_region_count,
            "validation_auprc": validation_auprc,
            "learning_rate": current_lr,
            "eta_class_0_mean": eta_class_means[0],
            "eta_class_1_mean": eta_class_means[1],
        }
        history.append(row)
        if epoch == 1 or epoch % 10 == 0 or epoch == config.epochs:
            print(
                f"  Encoder epoch {epoch:03d}/{config.epochs} | "
                f"loss={row['loss']:.4f} lr={current_lr:.2e} "
                f"val_AUPRC={validation_auprc:.4f} "
                f"best={best_score:.4f}@{best_epoch} "
                f"proto_regions={prototype_cache.valid_region_count}"
            )
        if (
            epoch >= config.minimum_epochs
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            print(
                f"  Encoder early stop at epoch {epoch}; "
                f"restore epoch {best_epoch} (validation AUPRC={best_score:.4f})."
            )
            break
    if best_state is None:
        raise RuntimeError("编码器没有产生可恢复的最佳epoch。")
    model.load_state_dict(best_state)
    for row in history:
        row["is_best_epoch"] = int(row["epoch"] == best_epoch)
    model.eval()
    return model, history


@torch.no_grad()
def generate_residual_features(
    model: ResidualInteractionEncoder,
    values: np.ndarray,
    masks: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outputs = []
    for start in range(0, len(values), batch_size):
        batch_values = torch.as_tensor(values[start:start + batch_size], dtype=torch.float32, device=device)
        batch_masks = torch.as_tensor(masks[start:start + batch_size], dtype=torch.float32, device=device)
        outputs.append(model.encode(batch_values, batch_masks).cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def choose_threshold(y_valid: np.ndarray, p_valid: np.ndarray) -> float:
    return youden_threshold(y_valid, p_valid)


def calculate_metrics(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float]:
    prediction = (probability >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "auprc": float(average_precision_score(y, probability)),
        "auroc": float(roc_auc_score(y, probability)),
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else float("nan"),
        "specificity": float(tn / (tn + fp)) if tn + fp else float("nan"),
        "brier": float(brier_score_loss(y, probability)),
        "threshold": float(threshold),
        "n_test": int(len(y)),
        "n_positive_test": int((y == 1).sum()),
    }


def summarize(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    names = PRIMARY_METRICS + DIAGNOSTIC_METRICS
    return pd.DataFrame([
        {
            "metric": name,
            "role": "primary" if name in PRIMARY_METRICS else "diagnostic",
            "mean": float(np.nanmean(fold_metrics[name].to_numpy(float))),
            "std": float(np.nanstd(fold_metrics[name].to_numpy(float), ddof=1)),
        }
        for name in names
    ])


def visible_gpu_tokens() -> list[str]:
    """Return physical/visible GPU tokens safe for child CUDA isolation."""
    count = torch.cuda.device_count()
    if count == 0:
        return []
    visible_env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible_env:
        available = [token.strip() for token in visible_env.split(",") if token.strip()]
    else:
        available = [str(index) for index in range(count)]
    available = available[:count]
    if GPU_IDS is None:
        return available
    selected = []
    for logical_id in GPU_IDS:
        if logical_id < 0 or logical_id >= len(available):
            raise ValueError(
                f"GPU_IDS包含不可用编号{logical_id}；当前可见GPU数量为{len(available)}。"
            )
        selected.append(available[logical_id])
    return selected


def write_final_outputs(
    fold_rows: list[dict[str, float]],
    prediction_rows: list[pd.DataFrame],
    data_path: Path,
    started: float,
    execution_mode: str,
    gpu_assignments: Optional[dict[str, list[int]]] = None,
) -> None:
    fold_frame = pd.DataFrame(fold_rows).sort_values("fold")
    predictions = pd.concat(prediction_rows, ignore_index=True).sort_values("row_index")
    summary = summarize(fold_frame)
    fold_frame.to_csv(OUTPUT_DIR / "fold_metrics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig")

    config = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "MedRIFT + XGBoost",
        "dataset": DATASET_NAME,
        "task": TASK_NAME,
        "data_file": str(data_path),
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "random_seed": RANDOM_SEED,
        "outer_folds": N_OUTER_FOLDS,
        "diagnostic_oof_folds": N_DIAGNOSTIC_FOLDS,
        "validation_fraction_of_trainval": VALIDATION_FRACTION_OF_TRAINVAL,
        "threshold_rule": "maximize Youden J on validation subset",
        "protocol_id": PROTOCOL_ID,
        "model_selection_rule": "all hyperparameters are frozen; validation selects only the encoder checkpoint, XGBoost stopping iteration, and decision threshold",
        "xgboost_early_stopping_metric": "negative sklearn average_precision_score; XGBoost minimizes the negative value, exactly maximizing the reported AP",
        "final_xgboost_max_estimators": FINAL_XGB_MAX_ESTIMATORS,
        "final_xgboost_early_stopping_rounds": FINAL_XGB_EARLY_STOPPING_ROUNDS,
        "primary_metrics": PRIMARY_METRICS,
        "diagnostic_outputs": DIAGNOSTIC_METRICS,
        "graph_file": str(GRAPH_PATH),
        "graph_rule": GRAPH_VARIANT,
        "preprocessing": "fit within each outer training subset; tree: numerical median and categorical one-hot without scaling; neural: numerical median plus Z-score and categorical embeddings; explicit missing masks",
        "encoder_protocol": {
            "name": encoder_config_for_dataset(DATASET_STEM)[0],
            **asdict(encoder_config_for_dataset(DATASET_STEM)[1]),
        },
        "encoder_training_enhancements": {
            "sample_weight_normalization":
                "eta_tilde_i = eta_i / mean_train(eta | class=y_i)",
            "learning_rate_schedule": "linear warmup followed by bounded cosine decay to 5% of base LR",
            "focal_loss": False,
            "label_smoothing": False,
            "mixup": False,
            "ema": False,
        },
        "xgboost_params": {
            **XGB_PARAMS,
            "n_estimators": FINAL_XGB_MAX_ESTIMATORS,
            "eval_metric": negative_average_precision.__name__,
        },
        "xgboost_fixed_candidate": XGB_FIXED_CANDIDATE,
        "diagnostic_probability_rule": "each unweighted diagnostic tree is fitted outside its held-out fold and stops on outer validation; the unweighted reference tree supplies leaf paths; the final tree uses training-fold class weight",
        "execution_mode": execution_mode,
        "gpu_assignments": gpu_assignments or {},
        "xgboost_gpu_requested": USE_GPU_FOR_XGBOOST,
        "xgboost_device_supported": xgboost_supports_device(),
        "versions": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
            "torch": torch.__version__,
        },
    }
    (OUTPUT_DIR / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\n完成。5折均值±标准差：")
    for row in summary.itertuples(index=False):
        print(f"{row.metric}: {row.mean:.4f} ± {row.std:.4f}")
    print(f"总耗时：{(time.time() - started) / 60:.1f}分钟")
    print(f"结果文件夹：{OUTPUT_DIR}")


def run_multi_gpu_parent(data_path: Path, started: float, gpu_tokens: list[str]) -> None:
    """Launch one Windows-safe process per GPU and aggregate fold outputs."""
    worker_count = min(len(gpu_tokens), N_OUTER_FOLDS)
    gpu_tokens = gpu_tokens[:worker_count]
    pending = [fold for fold in range(1, N_OUTER_FOLDS + 1)
               if not fold_is_complete(OUTPUT_DIR / f"fold{fold}")]
    fold_groups = [[] for _ in range(worker_count)]
    for fold in pending:
        fold_groups[(fold - 1) % worker_count].append(fold)

    assignments = {gpu_tokens[i]: fold_groups[i] for i in range(worker_count)}
    cpu_threads = CPU_THREADS_PER_WORKER
    if cpu_threads <= 0:
        cpu_threads = max(1, (os.cpu_count() or worker_count) // worker_count)

    print(f"检测到{len(gpu_tokens)}张可见GPU；待运行fold={pending}。")
    print(f"GPU分配：{assignments}")
    processes = []
    for token, folds in assignments.items():
        if not folds:
            continue
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = token
        environment["MEDRIFT_WORKER"] = "1"
        environment["MEDRIFT_FOLDS"] = ",".join(str(fold) for fold in folds)
        environment["MEDRIFT_CPU_THREADS"] = str(cpu_threads)
        environment["OMP_NUM_THREADS"] = str(cpu_threads)
        environment["MKL_NUM_THREADS"] = str(cpu_threads)
        environment["OPENBLAS_NUM_THREADS"] = str(cpu_threads)
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve())],
            cwd=str(BASE_DIR),
            env=environment,
        )
        processes.append((token, folds, process))

    failures = []
    for token, folds, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failures.append((token, folds, return_code))
    if failures:
        raise RuntimeError(f"多GPU worker运行失败：{failures}")

    fold_rows = []
    prediction_rows = []
    for fold in range(1, N_OUTER_FOLDS + 1):
        fold_dir = OUTPUT_DIR / f"fold{fold}"
        if not fold_is_complete(fold_dir):
            raise RuntimeError(f"Fold {fold}没有当前协议的完成标记。")
        metrics_path = fold_dir / "fold_metrics.json"
        predictions_path = fold_dir / "test_predictions.csv"
        if not metrics_path.exists() or not predictions_path.exists():
            raise RuntimeError(f"Fold {fold}未生成完整结果，拒绝汇总。")
        fold_rows.append(json.loads(metrics_path.read_text(encoding="utf-8")))
        prediction_rows.append(pd.read_csv(predictions_path, encoding="utf-8-sig"))
    write_final_outputs(
        fold_rows, prediction_rows, data_path, started,
        execution_mode="parallel_outer_folds_across_gpus",
        gpu_assignments=assignments,
    )


def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    worker_mode = os.environ.get("MEDRIFT_WORKER") == "1"
    dataset_stem = os.environ.get("MEDRIFT_DATASET", "").strip()
    if not dataset_stem and not worker_mode:
        if not DATA_DIR.exists():
            raise FileNotFoundError(f"找不到data文件夹：{DATA_DIR}")
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        print("将依次运行四个数据集；每个数据集内部会自动使用全部可见GPU分配外层fold。")
        for stem in DATASETS_TO_RUN:
            environment = os.environ.copy()
            environment["MEDRIFT_DATASET"] = stem
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).resolve())],
                cwd=str(BASE_DIR), env=environment,
            )
            if completed.returncode != 0:
                raise RuntimeError(f"数据集 {stem} 运行失败，已停止后续任务。")
        overview = []
        for stem in DATASETS_TO_RUN:
            summary_path = OUTPUT_ROOT / stem / "summary.csv"
            summary = pd.read_csv(summary_path, encoding="utf-8-sig")
            row = {"dataset": stem}
            for item in summary.itertuples(index=False):
                if item.metric in PRIMARY_METRICS:
                    row[f"{item.metric}_mean"] = item.mean
                    row[f"{item.metric}_std"] = item.std
            overview.append(row)
        pd.DataFrame(overview).to_csv(
            OUTPUT_ROOT / "all_datasets_overview.csv", index=False, encoding="utf-8-sig"
        )
        print(f"\n四个MedRIFT数据集全部完成：{OUTPUT_ROOT}")
        return
    if not dataset_stem:
        raise ValueError("Worker未收到MEDRIFT_DATASET。")
    if dataset_stem not in DATASETS_TO_RUN:
        raise ValueError(f"未知数据集：{dataset_stem}")
    configure_dataset(dataset_stem)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data_path = locate_data()
    X, y, patient_ids = read_data(data_path)
    graph_variant = GRAPH_VARIANT
    if graph_variant == "rewired":
        edge_index = None
        graph_provenance = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
        graph_provenance["active_graph_variant"] = "rewired_per_outer_fold"
    else:
        edge_index, graph_provenance = build_graph()
    encoder_name, encoder_config = encoder_config_for_dataset(DATASET_STEM)
    canonical_splits = build_or_validate_splits(DATASET_STEM)

    # The parent process only schedules folds. CUDA_VISIBLE_DEVICES isolates
    # every child to one physical GPU, which then appears locally as cuda:0.
    if not worker_mode and USE_ALL_GPUS:
        gpu_tokens = visible_gpu_tokens()
        if len(gpu_tokens) > 1:
            if graph_variant != "rewired":
                (OUTPUT_DIR / "medical_graph.json").write_text(
                    json.dumps(graph_provenance, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            del X, y, patient_ids, canonical_splits
            started = time.time()
            run_multi_gpu_parent(data_path, started, gpu_tokens)
            return

    set_seed(RANDOM_SEED)
    selected_folds = set(range(1, N_OUTER_FOLDS + 1))
    if worker_mode:
        requested = os.environ.get("MEDRIFT_FOLDS", "")
        selected_folds = {int(item) for item in requested.split(",") if item.strip()}
        if not selected_folds:
            raise ValueError("GPU worker没有收到任何fold编号。")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("MedRIFT + XGBoost开始。")
    print(f"数据：{data_path}")
    print(f"样本：{len(y)}；阴性：{int((y == 0).sum())}；阳性：{int((y == 1).sum())}")
    print(f"医学图：|E_med|={len(graph_provenance['E_med'])}，"
          f"|E_group|={len(graph_provenance['E_group'])}，"
          f"|E|={len(graph_provenance['E'])}（无相关性边）")
    if worker_mode:
        print(f"Worker folds={sorted(selected_folds)}；本地设备={device}")
    else:
        print(f"神经网络设备：{device}")
    print(f"输出：{OUTPUT_DIR}")

    if not worker_mode and graph_variant != "rewired":
        (OUTPUT_DIR / "medical_graph.json").write_text(
            json.dumps(graph_provenance, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    fold_rows: list[dict[str, float]] = []
    prediction_rows: list[pd.DataFrame] = []
    started = time.time()

    for fold, canonical in enumerate(canonical_splits, start=1):
        if fold not in selected_folds:
            continue
        completed_dir = OUTPUT_DIR / f"fold{fold}"
        if fold_is_complete(completed_dir):
            print(f"Fold {fold} 已完成，安全跳过。")
            if not worker_mode:
                fold_rows.append(json.loads((completed_dir / "fold_metrics.json").read_text(encoding="utf-8")))
                prediction_rows.append(pd.read_csv(completed_dir / "test_predictions.csv", encoding="utf-8-sig"))
            continue
        print(f"\n========== Fold {fold}/{N_OUTER_FOLDS} ==========")
        fold_started = time.perf_counter()
        train_idx, valid_idx, test_idx = canonical["train"], canonical["valid"], canonical["test"]
        train_frame = X.iloc[train_idx].reset_index(drop=True)
        valid_frame = X.iloc[valid_idx].reset_index(drop=True)
        y_train, y_valid = y[train_idx], y[valid_idx]
        fold_seed = RANDOM_SEED + 1000 * fold
        if graph_variant == "rewired":
            fold_edge_index, fold_graph_provenance = build_graph(fold=fold)
        else:
            fold_edge_index, fold_graph_provenance = edge_index, graph_provenance

        print("1/5 生成OOF残差诊断……")
        oof_positive = diagnostic_oof_probabilities(
            train_frame, y_train, valid_frame, y_valid, fold_seed,
        )
        true_class_probability = np.where(y_train == 1, oof_positive, 1.0 - oof_positive)
        residuals = (1.0 - true_class_probability).astype(np.float32)

        print("2/5 拟合参考树并构建树区域……")
        tree_preprocessor = make_tree_preprocessor()
        tree_train = tree_preprocessor.fit_transform(train_frame).astype(np.float32)
        tree_valid = tree_preprocessor.transform(valid_frame).astype(np.float32)
        reference_model, selected_base, base_tree_trials = fit_fixed_xgboost(
            tree_train, y_train, tree_valid, y_valid, fold_seed + 100,
            use_class_weight=False,
        )
        region_config = encoder_config
        regions, ambiguity, leaf_encoder, svd, kmeans = build_tree_regions(
            reference_model, tree_train, y_train, region_config, fold_seed + 200
        )

        print("3/5 训练固定配置的医学知识约束残差交互编码器……")
        neural_preprocessor = NeuralPreprocessor().fit(train_frame)
        neural_train, mask_train = neural_preprocessor.transform(train_frame)
        neural_valid, mask_valid = neural_preprocessor.transform(valid_frame)
        selected_encoder_name = encoder_name
        selected_encoder_config = encoder_config
        encoder, history = train_encoder(
            neural_train, mask_train, neural_valid, mask_valid,
            y_train, y_valid, residuals, regions, ambiguity,
            neural_preprocessor.category_cardinalities(), fold_edge_index,
            selected_encoder_config, device, fold_seed + 300,
        )

        print("4/5 生成残差特征并训练固定配置最终XGBoost……")
        z_train = generate_residual_features(
            encoder, neural_train, mask_train, device, selected_encoder_config.batch_size
        )
        z_valid = generate_residual_features(
            encoder, neural_valid, mask_valid, device, selected_encoder_config.batch_size
        )
        enhanced_train = np.concatenate([tree_train, z_train], axis=1)
        enhanced_valid = np.concatenate([tree_valid, z_valid], axis=1)
        final_model, selected_final, final_tree_trials = fit_fixed_xgboost(
            enhanced_train, y_train, enhanced_valid, y_valid,
            RANDOM_SEED + fold, use_class_weight=True,
        )
        training_time_seconds = time.perf_counter() - fold_started

        print("5/5 在验证集定阈值，并评估未见测试折……")
        test_frame = X.iloc[test_idx].reset_index(drop=True)
        y_test = y[test_idx]
        tree_test = tree_preprocessor.transform(test_frame).astype(np.float32)
        neural_test, mask_test = neural_preprocessor.transform(test_frame)
        z_test = generate_residual_features(
            encoder, neural_test, mask_test, device, selected_encoder_config.batch_size
        )
        enhanced_test = np.concatenate([tree_test, z_test], axis=1)
        p_valid = final_model.predict_proba(enhanced_valid)[:, 1]
        p_test = final_model.predict_proba(enhanced_test)[:, 1]
        p_diagnostic_valid = reference_model.predict_proba(tree_valid)[:, 1]
        p_diagnostic_test = reference_model.predict_proba(tree_test)[:, 1]
        diagnostic_true_probability = np.where(y_test == 1, p_diagnostic_test, 1.0 - p_diagnostic_test)
        diagnostic_test_residual = 1.0 - diagnostic_true_probability
        threshold = choose_threshold(y_valid, p_valid)
        diagnostic_threshold = choose_threshold(y_valid, p_diagnostic_valid)
        metrics = calculate_metrics(y_test, p_test, threshold)
        best_encoder_row = max(history, key=lambda item: item["validation_auprc"])
        metrics.update({
            "fold": fold,
            "n_train": int(len(train_idx)),
            "n_valid": int(len(valid_idx)),
            "mean_residual": float(residuals.mean()),
            "mean_region_ambiguity": float(ambiguity[regions].mean()),
            "selected_base_tree_candidate": selected_base["candidate"],
            "base_tree_best_iteration": int(selected_base["best_iteration"]),
            "selected_encoder_candidate": selected_encoder_name,
            "best_encoder_epoch": int(best_encoder_row["epoch"]),
            "best_encoder_validation_auprc": float(best_encoder_row["validation_auprc"]),
            "valid_prototype_regions_best_epoch": int(best_encoder_row["valid_prototype_regions"]),
            "selected_final_tree_candidate": selected_final["candidate"],
            "final_tree_validation_auprc": float(selected_final["validation_auprc"]),
            "best_xgboost_iteration": int(selected_final["best_iteration"]),
            "training_time_seconds": training_time_seconds,
            "encoder_parameters": int(sum(p.numel() for p in encoder.parameters() if p.requires_grad)),
            "residual_dimension": int(selected_encoder_config.residual_dim),
            "experiment_variant": EXPERIMENT_VARIANT,
        })
        fold_rows.append(metrics)
        prediction_rows.append(pd.DataFrame({
            "patient_id": patient_ids[test_idx],
            "row_index": test_idx,
            "fold": fold,
            "y_true": y_test,
            "probability": p_test,
            "threshold": threshold,
            "prediction": (p_test >= threshold).astype(np.int64),
            "diagnostic_probability": p_diagnostic_test,
            "diagnostic_threshold": diagnostic_threshold,
            "diagnostic_prediction": (p_diagnostic_test >= diagnostic_threshold).astype(np.int64),
            "diagnostic_residual": diagnostic_test_residual,
        }))

        fold_dir = OUTPUT_DIR / f"fold{fold}"
        fold_dir.mkdir(exist_ok=True)
        np.savez_compressed(
            fold_dir / "split_indices.npz",
            train=train_idx, valid=valid_idx, test=test_idx,
        )
        pd.DataFrame(history).to_csv(fold_dir / "encoder_history.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([{
            "protocol": selected_encoder_name,
            "best_epoch": best_encoder_row["epoch"],
            "validation_auprc": best_encoder_row["validation_auprc"],
            **asdict(selected_encoder_config),
        }]).to_csv(fold_dir / "encoder_protocol.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(base_tree_trials).to_csv(
            fold_dir / "base_tree_protocol.csv", index=False, encoding="utf-8-sig"
        )
        pd.DataFrame(final_tree_trials).to_csv(
            fold_dir / "final_tree_protocol.csv", index=False, encoding="utf-8-sig"
        )
        np.savez_compressed(
            fold_dir / "diagnostics.npz",
            train_row_index=train_idx,
            residual_score=residuals,
            tree_region=regions,
            region_ambiguity=ambiguity,
        )
        torch.save(
            {
                "state_dict": encoder.state_dict(),
                "category_cardinalities": neural_preprocessor.category_cardinalities(),
                "edge_index": torch.as_tensor(fold_edge_index, dtype=torch.long),
                "config": asdict(selected_encoder_config),
                "candidate": selected_encoder_name,
            },
            fold_dir / "residual_encoder.pt",
        )
        (fold_dir / "medical_graph.json").write_text(
            json.dumps(fold_graph_provenance, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        joblib.dump(
            {
                "tree_preprocessor": tree_preprocessor,
                "neural_preprocessor_state": {
                    "medians": neural_preprocessor.medians,
                    "means": neural_preprocessor.means,
                    "stds": neural_preprocessor.stds,
                    "categories": neural_preprocessor.categories,
                },
                "reference_model": reference_model,
                "selected_base_tree": selected_base,
                "leaf_encoder": leaf_encoder,
                "svd": svd,
                "kmeans": kmeans,
                "final_model": final_model,
                "selected_final_tree": selected_final,
                "threshold": threshold,
            },
            fold_dir / "non_torch_models.joblib",
            compress=3,
        )
        # Per-fold completion files are the only objects read by the parent.
        # They are written after all fold models and diagnostics succeed.
        (fold_dir / "fold_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        prediction_rows[-1].to_csv(
            fold_dir / "test_predictions.csv", index=False, encoding="utf-8-sig"
        )
        (fold_dir / "SUCCESS.json").write_text(
            json.dumps({"status": "success", "variant": EXPERIMENT_VARIANT,
                        "protocol_id": PROTOCOL_ID}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"Fold {fold}完成 | AUPRC={metrics['auprc']:.4f} "
            f"AUROC={metrics['auroc']:.4f} threshold={threshold:.4f}"
        )

        # Prevent retained GPU tensors from accumulating across folds.
        del encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if worker_mode:
        print(f"Worker folds={sorted(selected_folds)}全部完成。")
        return
    write_final_outputs(
        fold_rows, prediction_rows, data_path, started,
        execution_mode="serial_single_gpu_or_cpu",
    )


if __name__ == "__main__":
    main()
