"""Pure scoring functions. No I/O, no agent code touches this."""
from __future__ import annotations

import numpy as np


def dice(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    """Dice coefficient over a batch of binary masks, shape (N,H,W) or (H,W)."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float((2.0 * intersection + eps) / (denom + eps))


def iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7) -> float:
    """Intersection-over-union over a batch of binary masks."""
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float((intersection + eps) / (union + eps))
