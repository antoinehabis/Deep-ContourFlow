"""Segmentation / foreground-extraction metrics.

All functions take binary masks (0/1, any shape) as numpy arrays and compare a
prediction against a ground truth. ``SegmentationMetrics`` accumulates the
per-sample scores across batches and returns their averages.

Metrics
-------
iou                 Intersection-over-Union (Jaccard).
dice                Dice coefficient == F1 of the foreground pixels.
pixel_accuracy      Fraction of correctly classified pixels.
f_measure           F-β score (default β=0.3 for salient object detection).
mae                 Mean Absolute Error (pixel-wise absolute difference).
e_measure           Edge-aware metric (boundary quality).
s_measure           Structure-aware metric (spatial consistency).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import scipy.ndimage as ndimage

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


def f_measure(pred: np.ndarray, gt: np.ndarray, beta: float = 0.3) -> float:
    """F-β score (default β=0.3 for salient object detection)."""
    pred, gt = _binarize(pred), _binarize(gt)
    tp = float(np.sum(pred * gt))
    fp = float(np.sum(pred * (1 - gt)))
    fn = float(np.sum((1 - pred) * gt))

    precision = tp / (tp + fp + EPS)
    recall = tp / (tp + fn + EPS)

    beta_sq = beta ** 2
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall + EPS)


def mae(pred: np.ndarray, gt: np.ndarray) -> float:
    """Mean Absolute Error: average pixel-wise absolute difference."""
    pred, gt = np.asarray(pred), np.asarray(gt)
    # Normalize to [0, 1]
    pred = np.clip(pred, 0, 1)
    gt = _binarize(gt)
    return float(np.mean(np.abs(pred - gt)))


def e_measure(pred: np.ndarray, gt: np.ndarray) -> float:
    """Edge-aware metric measuring boundary quality.

    Computes enhanced-alignment measure (E-measure) that evaluates
    how well predicted boundaries align with ground truth edges.
    """
    pred, gt = np.asarray(pred, dtype=np.float32), np.asarray(gt, dtype=np.float32)
    pred = np.clip(pred, 0, 1)
    gt = _binarize(gt)

    # Compute edge maps
    gy, gx = np.gradient(gt)
    gt_edge = (np.sqrt(gx**2 + gy**2) > 0.0)

    fy, fx = np.gradient(pred)
    pred_edge = (np.sqrt(fx**2 + fy**2) > 0.0)

    # Alignment between prediction and gt
    if not np.any(gt_edge) and not np.any(pred_edge):
        return 1.0

    # Compute mean alignment
    align = 2 * np.sum(pred_edge & gt_edge) / (float(np.sum(pred_edge)) + float(np.sum(gt_edge)) + EPS)

    # Regional overlap
    inter = float(np.sum(pred * gt))
    union = float(np.sum(pred) + np.sum(gt) - inter)
    region = inter / (union + EPS)

    return 0.5 * align + 0.5 * region


def s_measure(pred: np.ndarray, gt: np.ndarray) -> float:
    """Structure-aware metric measuring spatial/structural consistency.

    Evaluates how well the predicted mask preserves spatial structure
    and object connectivity of the ground truth.
    """
    pred, gt = np.asarray(pred, dtype=np.float32), np.asarray(gt, dtype=np.float32)
    pred = np.clip(pred, 0, 1)
    gt = _binarize(gt)

    # Object region overlap
    inter = float(np.sum(pred * gt))
    union = float(np.sum(np.maximum(pred, gt)))
    iou_val = inter / (union + EPS)

    # Structure preservation: connectivity analysis
    gt_labeled, gt_num = ndimage.label(gt > 0.5)
    pred_labeled, pred_num = ndimage.label(pred > 0.5)

    # Measure overlap of largest connected components
    if gt_num > 0 and pred_num > 0:
        gt_largest = np.sum(gt_labeled == 1)
        pred_largest = np.sum(pred_labeled == 1)
        overlap = np.sum((gt_labeled == 1) & (pred_labeled == 1))
        struct_sim = float(overlap) / float(max(gt_largest, pred_largest, 1))
    else:
        struct_sim = float(gt_num == pred_num)

    return 0.5 * iou_val + 0.5 * struct_sim


# Registry of per-sample scalar metrics.
METRICS = {
    "iou": iou,
    "dice": dice,
    "pixel_accuracy": pixel_accuracy,
    "f_measure": f_measure,
    "mae": mae,
    "e_measure": e_measure,
    "s_measure": s_measure,
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
