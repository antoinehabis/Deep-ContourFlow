"""Segmentation / foreground-extraction metrics.

All functions take binary masks (0/1, any shape) as numpy arrays and compare a
prediction against a ground truth. ``SegmentationMetrics`` accumulates the
per-sample scores across batches and returns their averages.

Metrics
-------
iou            Intersection-over-Union (Jaccard).
dice           Dice coefficient == F1 of the foreground pixels.
pixel_accuracy Fraction of correctly classified pixels.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

EPS = 1e-7


def _binarize(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (np.asarray(mask) > threshold).astype(np.float64)


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = _binarize(pred), _binarize(gt)
    inter = float(np.sum(pred * gt))
    union = float(np.sum(pred) + np.sum(gt) - inter)
    return inter / (union + EPS)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = _binarize(pred), _binarize(gt)
    inter = float(np.sum(pred * gt))
    return (2 * inter) / (float(np.sum(pred) + np.sum(gt)) + EPS)


def pixel_accuracy(pred: np.ndarray, gt: np.ndarray) -> float:
    pred, gt = _binarize(pred), _binarize(gt)
    correct = float(np.sum(pred == gt))
    return correct / (pred.size + EPS)


# Registry of per-sample scalar metrics.
METRICS = {
    "iou": iou,
    "dice": dice,
    "pixel_accuracy": pixel_accuracy,
}


def compute_all(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    """All metrics for a single (pred, gt) pair."""
    return {name: fn(pred, gt) for name, fn in METRICS.items()}


class SegmentationMetrics:
    """Accumulate per-sample metrics across a run and report the mean."""

    def __init__(self):
        self._sums: Dict[str, float] = {name: 0.0 for name in METRICS}
        self.n = 0

    def update(self, preds, gts) -> None:
        """Add a batch. ``preds``/``gts`` are iterables of 2D binary masks."""
        for pred, gt in zip(preds, gts):
            pred = np.asarray(pred)
            gt = np.asarray(gt)
            for name, fn in METRICS.items():
                self._sums[name] += fn(pred, gt)
            self.n += 1

    def average(self) -> Dict[str, float]:
        if self.n == 0:
            return {name: float("nan") for name in METRICS}
        return {name: self._sums[name] / self.n for name in METRICS}

    def __len__(self) -> int:
        return self.n
