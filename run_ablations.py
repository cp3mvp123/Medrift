# 此文件更新于 2026-08-20 15:26（北京时间）
"""Run the core, graph, and parameter-sensitivity experiments in the paper."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from experiment_config import CORE_ABLATIONS, DATASETS, GRAPH_ABLATIONS, SENSITIVITY


ROOT = Path(__file__).resolve().parent


def run_variant(
    name: str,
    *,
    graph: str = "full",
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    prototype_weight: Optional[float] = None,
) -> None:
    for dataset in DATASETS:
        environment = os.environ.copy()
        for key in ("MEDRIFT_BETA", "MEDRIFT_GAMMA", "MEDRIFT_LAMBDA", "MEDRIFT_K", "MEDRIFT_DZ"):
            environment.pop(key, None)
        environment.update({
            "MEDRIFT_DATASET": dataset,
            "MEDRIFT_VARIANT": name,
            "MEDRIFT_GRAPH": graph,
        })
        if beta is not None:
            environment["MEDRIFT_BETA"] = str(beta)
        if gamma is not None:
            environment["MEDRIFT_GAMMA"] = str(gamma)
        if prototype_weight is not None:
            environment["MEDRIFT_LAMBDA"] = str(prototype_weight)
        print(f"START {name}: dataset={dataset}, graph={graph}")
        completed = subprocess.run(
            [sys.executable, str(ROOT / "medrift_engine.py")],
            cwd=ROOT,
            env=environment,
        )
        if completed.returncode:
            raise RuntimeError(
                f"Experiment failed: variant={name}, dataset={dataset}, "
                f"returncode={completed.returncode}"
            )


def main() -> None:
    for variant in CORE_ABLATIONS:
        run_variant(variant)
    for graph in GRAPH_ABLATIONS:
        run_variant(f"graph_{graph}", graph=graph)
    for value in SENSITIVITY["beta"]:
        run_variant(f"sensitivity_beta_{value:g}", beta=value)
    for value in SENSITIVITY["gamma"]:
        run_variant(f"sensitivity_gamma_{value:g}", gamma=value)
    for value in SENSITIVITY["lambda"]:
        run_variant(f"sensitivity_lambda_{value:g}", prototype_weight=value)


if __name__ == "__main__":
    main()
