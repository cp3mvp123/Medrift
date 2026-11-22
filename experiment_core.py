# 此文件更新于 2026-08-20 15:26（北京时间）
"""Leakage-safe data, split, metric and persistence utilities."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split

from experiment_config import (DATA_DIR, DATASETS, GRAPH_DIR, MANIFEST_DIR,
                               N_OUTER_FOLDS, PROTOCOL_ID, RANDOM_SEED,
                               SPLIT_DIR, VALIDATION_FRACTION_OF_TRAINVAL)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(sanitize_json(payload), ensure_ascii=False, indent=2, default=json_default, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, Path): return str(value)
    raise TypeError(type(value).__name__)


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict): return {str(k): sanitize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [sanitize_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value): return None
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def load_manifest(stem: str) -> dict[str, Any]:
    path = MANIFEST_DIR / f"{stem}_manifest.json"
    if not path.exists(): raise FileNotFoundError(f"缺少 manifest：{path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_dataset(stem: str, path: Optional[Path] = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    manifest = load_manifest(stem)
    path = DATA_DIR / f"{stem}.csv" if path is None else path
    if not path.exists(): raise FileNotFoundError(f"缺少处理后数据：{path}")
    id_col = manifest["id_column"]
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False, dtype={id_col: "string"})
    nums = list(manifest["continuous_features"]); cats = list(manifest["categorical_features"])
    features = nums + cats
    required = set(features + [id_col, manifest["target_column"]])
    missing = sorted(required - set(frame.columns))
    if missing: raise KeyError(f"{stem} 缺少字段：{missing}")
    converted = {
        col: pd.to_numeric(frame[col], errors="coerce") for col in nums
    }
    converted.update({
        col: frame[col].astype("object").where(frame[col].notna(), np.nan)
        for col in cats
    })
    X = pd.DataFrame(converted, index=frame.index)
    masks = {}
    for feature in features:
        mask_name = f"{feature}__missing"
        missing = X[feature].isna()
        if mask_name in frame.columns:
            supplied = pd.to_numeric(frame[mask_name], errors="coerce")
            if not supplied.isin([0, 1]).all():
                raise ValueError(f"{stem} {mask_name} 必须只包含0和1。")
            missing = missing | supplied.eq(1)
            X[feature] = X[feature].mask(missing)
        masks[mask_name] = missing.astype(np.int8)
    X = pd.concat([X, pd.DataFrame(masks, index=frame.index)], axis=1)
    labels = pd.to_numeric(frame[manifest["target_column"]], errors="raise")
    if set(labels.unique()) != {0, 1}: raise ValueError(f"{stem} 标签必须同时包含0和1。")
    y = labels.to_numpy(np.int64)
    ids = frame[id_col].str.strip()
    if ids.isna().any() or ids.eq("").any():
        raise ValueError(f"{stem} 患者ID不能为空。")
    if ids.duplicated().any():
        raise ValueError(f"{stem} 患者ID存在重复；必须先明确纵向记录并改用group split。")
    return X, y, ids.to_numpy(dtype=str), manifest


def build_or_validate_splits(stem: str) -> list[dict[str, np.ndarray]]:
    X, y, ids, _ = load_dataset(stem)
    out = SPLIT_DIR / stem; out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "split_manifest.json"
    expected = {"dataset": stem, "n": len(y), "id_hash": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
                "label_hash": hashlib.sha256(y.tobytes()).hexdigest(), "seed": RANDOM_SEED,
                "outer_folds": N_OUTER_FOLDS, "validation_fraction_of_trainval": VALIDATION_FRACTION_OF_TRAINVAL}
    if meta_path.exists() and json.loads(meta_path.read_text(encoding="utf-8")) != expected:
        raise RuntimeError(f"{stem} 数据内容变化，但旧划分仍存在。请人工确认后删除 {out}，禁止静默重划分。")
    if not meta_path.exists():
        splitter = StratifiedKFold(N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        for fold, (trainval, test) in enumerate(splitter.split(X, y), 1):
            train, valid = train_test_split(trainval, test_size=VALIDATION_FRACTION_OF_TRAINVAL,
                                            stratify=y[trainval], random_state=RANDOM_SEED + fold)
            np.savez_compressed(out / f"fold{fold}.npz", train=train, valid=valid, test=test)
        atomic_json(meta_path, expected)
    splits = []
    seen = np.zeros(len(y), dtype=int)
    for fold in range(1, N_OUTER_FOLDS + 1):
        item = dict(np.load(out / f"fold{fold}.npz"))
        sets = [set(item[k].tolist()) for k in ("train", "valid", "test")]
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]: raise RuntimeError("划分集合重叠。")
        for key in ("train", "valid", "test"):
            if len(np.unique(y[item[key]])) != 2: raise RuntimeError(f"{stem} fold{fold} {key} 缺少类别。")
        seen[item["test"]] += 1; splits.append(item)
    if not np.all(seen == 1): raise RuntimeError(f"{stem} 外层测试折没有恰好覆盖每位患者一次。")
    return splits


def negative_average_precision(y_true: np.ndarray, y_probability: np.ndarray) -> float:
    return -float(average_precision_score(y_true, y_probability))


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y, p)
    finite = np.isfinite(thresholds)
    if not finite.any(): return 0.5
    return float(thresholds[finite][np.argmax(tpr[finite] - fpr[finite])])


def metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    p = np.asarray(p, dtype=float)
    if p.shape != y.shape or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("预测概率形状错误、非有限或超出[0,1]。")
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"auprc": float(average_precision_score(y, p)), "auroc": float(roc_auc_score(y, p)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "brier": float(brier_score_loss(y, p)), "threshold": float(threshold),
            "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
            "specificity": float(tn / (tn + fp)) if tn + fp else np.nan}


def save_fold(out: Path, metric_row: dict[str, Any], predictions: pd.DataFrame, config: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out / "test_predictions.csv", index=False, encoding="utf-8-sig")
    atomic_json(out / "metrics.json", metric_row); atomic_json(out / "config.json", config)
    if (out / "FAILED.json").exists():
        (out / "FAILED.json").unlink()
    atomic_json(out / "SUCCESS.json", {
        "status": "success",
        "protocol_id": PROTOCOL_ID,
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    })


def fold_is_complete(out: Path) -> bool:
    """Resume only artifacts produced by the exact current protocol."""
    success = out / "SUCCESS.json"
    if not success.exists():
        return False
    try:
        payload = json.loads(success.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"完成标记损坏：{success}") from exc
    observed = payload.get("protocol_id")
    if observed != PROTOCOL_ID:
        raise RuntimeError(
            f"检测到旧实验协议结果：{out}（{observed!r} != {PROTOCOL_ID!r}）。"
            "请删除整个 results 文件夹后重新运行，禁止混合论文结果。"
        )
    return True


@contextmanager
def failure_guard(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    try: yield
    except Exception as exc:
        atomic_json(out / "FAILED.json", {"type": type(exc).__name__, "message": str(exc), "python": sys.version})
        raise


def environment_info() -> dict[str, Any]:
    result = {"python": sys.version, "platform": platform.platform(), "machine": platform.machine()}
    try:
        import torch
        result.update({"torch": torch.__version__, "cuda": torch.version.cuda,
                       "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
    except Exception as exc: result["torch_error"] = str(exc)
    return result


def validate_all() -> pd.DataFrame:
    rows = []
    for stem in DATASETS:
        X, y, _, manifest = load_dataset(stem)
        graph_path = GRAPH_DIR / f"{stem}_graph.json"
        graph = json.loads(graph_path.read_text(encoding="utf-8"))
        features = list(manifest["continuous_features"]) + list(manifest["categorical_features"])
        if set(graph["nodes"]) != set(features): raise ValueError(f"{stem} 图节点与输入特征不一致。")
        for key in ("E_med", "E_group", "E"):
            edges = [tuple(sorted(e)) for e in graph[key]]
            if len(edges) != len(set(edges)) or any(a == b for a, b in edges): raise ValueError(f"{stem} {key} 有重复边或自环。")
        union = {tuple(sorted(e)) for e in graph["E_med"]} | {tuple(sorted(e)) for e in graph["E_group"]}
        if union != {tuple(sorted(e)) for e in graph["E"]}: raise ValueError(f"{stem} E != E_med union E_group。")
        build_or_validate_splits(stem)
        rows.append({"dataset": stem, "samples": len(y), "positives": int(y.sum()), "positive_rate": y.mean(),
                     "features": len(features), "numerical": len(manifest["continuous_features"]),
                     "categorical": len(manifest["categorical_features"]), "missing_rate": float(X[features].isna().mean().mean()),
                     "E_med": len(graph["E_med"]), "E_group": len(graph["E_group"]), "E": len(graph["E"])})
    return pd.DataFrame(rows)
