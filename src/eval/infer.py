"""Deterministic, package-level inference + evaluation.

infer() and evaluate() are never written by an agent -- they are the fixed contract
every agent's generated pipeline.py must feed correctly into (see
agents/pipeline_contract.py for the guardrail that checks this before a script is
trusted to reach evaluate()).
"""
from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np

from eval.metrics import dice, iou


def infer(pipeline_module: ModuleType, model: Any, X_raw: np.ndarray) -> np.ndarray:
    """Run the agent's preprocess() then the trained model's predict()."""
    X_processed = pipeline_module.preprocess(X_raw)
    preds = model.predict(X_processed)
    return np.asarray(preds).astype(np.uint8)


def evaluate(
    pipeline_module: ModuleType, model: Any, X: np.ndarray, y: np.ndarray
) -> dict[str, float]:
    """Run infer() then score the result against ground truth y."""
    preds = infer(pipeline_module, model, X)
    return {"dice": dice(preds, y), "iou": iou(preds, y)}
