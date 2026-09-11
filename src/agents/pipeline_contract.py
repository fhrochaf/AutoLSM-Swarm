"""The fixed interface every agent-generated pipeline.py must implement.

This is the guardrail: before a script's model is trusted to reach eval/infer.py,
validate_module() checks the required functions exist with the right shape, and
smoke_test() actually runs the full preprocess -> build_model -> train -> predict
chain on a couple of tiles to catch runtime errors early and cheaply.
"""
from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np

REQUIRED_FUNCS = ("preprocess", "build_model", "train")


class ContractError(Exception):
    """Raised when a generated pipeline module violates the required contract."""


def validate_module(mod: ModuleType) -> None:
    """Static checks: required functions exist, are callable, train() is documented."""
    errors = []
    for name in REQUIRED_FUNCS:
        func = getattr(mod, name, None)
        if func is None:
            errors.append(f"missing required function `{name}(...)`")
        elif not callable(func):
            errors.append(f"`{name}` exists but is not callable")

    train = getattr(mod, "train", None)
    if callable(train) and not (train.__doc__ and train.__doc__.strip()):
        errors.append(
            "`train(...)` must have a docstring -- if the method needs no training, "
            "the docstring must say why (e.g. a fixed-threshold heuristic)"
        )

    if errors:
        raise ContractError("Pipeline contract violated:\n- " + "\n- ".join(errors))


def smoke_test(mod: ModuleType, X_sample: np.ndarray, y_sample: np.ndarray) -> Any:
    """Run the full chain on a small sample. Returns the trained model on success.

    Raises ContractError with a descriptive message on any failure -- this message
    is what gets fed back to the LLM in the debug loop.
    """
    try:
        X_processed = mod.preprocess(X_sample)
    except Exception as exc:  # noqa: BLE001 - deliberately broad, re-raised as ContractError
        raise ContractError(f"preprocess() raised: {exc!r}") from exc

    try:
        model = mod.build_model(None)
    except Exception as exc:  # noqa: BLE001
        raise ContractError(f"build_model(None) raised: {exc!r}") from exc

    try:
        model = mod.train(model, X_processed, y_sample)
    except Exception as exc:  # noqa: BLE001
        raise ContractError(f"train(...) raised: {exc!r}") from exc

    if not hasattr(model, "predict") or not callable(model.predict):
        raise ContractError(
            "the object returned by train(...) has no callable .predict(...) method"
        )

    try:
        preds = np.asarray(model.predict(X_processed))
    except Exception as exc:  # noqa: BLE001
        raise ContractError(f"model.predict(...) raised: {exc!r}") from exc

    if preds.shape != y_sample.shape:
        raise ContractError(
            f"model.predict(...) returned shape {preds.shape}, "
            f"expected {y_sample.shape} (one binary mask per input tile)"
        )

    unique_vals = set(np.unique(preds).tolist())
    if not unique_vals <= {0, 1}:
        raise ContractError(
            f"model.predict(...) must return binary {{0,1}} masks, got values {unique_vals}. "
            "Threshold/binarize inside predict() before returning."
        )

    return model


REQUIRED_FUNCS_DOC = (
    "preprocess(X: np.ndarray[N,128,128,14]) -> np.ndarray  "
    "(prepare the raw data however your method needs -- feature engineering, "
    "data/sensor fusion, band/predictor selection, normalization, a statistical "
    "transform of the predictors, or a plain pass-through if your method needs none "
    "of that. Whatever it does, it must be applied identically at train and "
    "inference time.)\n"
    "build_model(config: dict | None) -> Any  (return an untrained instance of "
    "whatever predictive object your method uses -- a trainable classifier/model, a "
    "statistical model to be fitted, a physically-based or rule-based method with "
    "parameters to calibrate, etc.)\n"
    "train(model, X_processed, y: np.ndarray[N,128,128]) -> Any  "
    "(fit/calibrate the model and return it; must have a non-empty docstring. If "
    "your method needs no training or calibration -- e.g. a fixed-threshold or "
    "purely physically-based method -- train() may be a no-op, but the docstring "
    "must say why.)\n"
    "Whatever kind of method it is, the returned model must implement "
    ".predict(X_processed) -> np.ndarray[N,128,128] with binary {0,1} values."
)
