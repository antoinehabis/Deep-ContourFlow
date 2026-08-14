"""Knee / elbow detection on the DCF energy curve.

Pure-numpy (no torch / matplotlib) so that both the algorithm
(``unsupervised.DCF._compute_final_contours``) and the visualisation utilities
share a single, consistent implementation — the contour drawn as the "knee"
panel is then guaranteed to be the one ``DCF.predict`` returns.

DCF energy curves are often a *staircase*: several descent phases separated by
plateaus where the contour momentarily settles (on the object boundary, then on
sub-parts as a shrink force keeps pulling it in). The energy keeps decreasing the
whole time, so "end of the descent" is meaningless — what matters are the
*plateaus*. :func:`knee_index` finds them and returns the ONSET of the last one
(the last "coude": where the curve stops descending and goes flat). It ignores
any trailing descent that never settles (over-shrink / collapse), and degrades to
the single global elbow when there is no plateau at all.
"""

from __future__ import annotations

import numpy as np


def _smooth(y: np.ndarray) -> np.ndarray:
    """Box moving-average that PRESERVES the endpoints.

    ``np.convolve(..., mode="same")`` zero-pads the boundaries, pulling the
    first/last samples toward 0 and distorting slopes at the ends. We edge-pad
    (replicate the boundary value) instead, so ys[0]≈y[0], ys[-1]≈y[-1].
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    k = min(5, n)
    if k < 2:
        return y
    left, right = k // 2, k - 1 - k // 2
    yp = np.pad(y, (left, right), mode="edge")
    return np.convolve(yp, np.ones(k) / k, mode="valid")  # length == n


def _norm_xy(y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Normalise a curve to the unit box. Returns (x in [0,1], y in [0,1], y-range)."""
    n = len(y)
    xg = np.arange(n, dtype=float) / max(n - 1, 1)
    yr = float(y.max() - y.min())
    yg = (y - y.min()) / (yr + 1e-12)
    return xg, yg, yr


def _chord_vdist(xg: np.ndarray, yg: np.ndarray) -> np.ndarray:
    """Vertical distance of each point to the first->last chord (unit-box y)."""
    w = xg[-1] - xg[0]
    if w < 1e-12:
        return np.zeros_like(yg)
    chord = yg[0] + (yg[-1] - yg[0]) * (xg - xg[0]) / w
    return np.abs(yg - chord)


def knee_index(
    loss: np.ndarray,
    flat_frac: float = 0.15,
    min_drop: float = 0.05,
    min_plateau: int = 3,
    which: str = "last",
) -> int:
    """Onset of the last plateau ("coude") on an energy curve.

    Smooth the curve, normalise to the unit box, and mark the points whose slope
    is below ``flat_frac`` of the steepest slope — those are the *flat* points.
    Contiguous flat runs of length >= ``min_plateau`` that are preceded by a real
    descent (>= ``min_drop`` of the total range since the previous plateau) are
    genuine plateaus. Return the onset (first index) of the last one (``which=
    "last"``) or the first (``which="first"``). Falls back to the single
    chord-distance elbow if there is no plateau.

    Parameters
    ----------
    loss        : 1-D energy per frame. NaNs are dropped; the caller must trim
                  trailing padding (e.g. early-stop zeros) so only real frames
                  are passed.
    flat_frac   : a point is "flat" if its slope is below this fraction of the
                  steepest slope. Larger => plateaus start earlier / are looser.
    min_drop    : a plateau must follow a descent of at least this fraction of
                  the total range (filters spurious flats that aren't real
                  settling points).
    min_plateau : minimum length (frames) of a flat run to count as a plateau.
    which       : "last" (default) or "first" plateau onset.

    Returns
    -------
    int index into the (NaN-filtered) ``loss`` array.
    """
    y = np.asarray(loss, dtype=float)
    y = y[~np.isnan(y)]
    n = len(y)
    if n < min_plateau + 1:
        return max(0, n - 1)

    ys = _smooth(y)
    xg, yg, yr = _norm_xy(ys)
    if yr <= 1e-12:
        return n - 1

    slope = -np.gradient(yg, xg)             # positive on descents
    steepest = float(slope.max())
    if steepest <= 1e-12:
        return n - 1
    flat = slope < flat_frac * steepest

    # Contiguous flat runs.
    runs = []
    i = 0
    while i < n:
        if flat[i]:
            j = i
            while j + 1 < n and flat[j + 1]:
                j += 1
            runs.append((i, j))
            i = j + 1
        else:
            i += 1

    # Keep runs that are long enough AND preceded by a real descent since the
    # last accepted plateau (ref) — so trailing descent that never settles and
    # tiny noise-flats are ignored.
    plateaus = []
    ref = 0
    for a, b in runs:
        if (b - a + 1) >= min_plateau and (yg[ref] - yg[a]) >= min_drop:
            plateaus.append((a, b))
            ref = b

    if not plateaus:
        return int(np.argmax(_chord_vdist(xg, yg)))
    a, _ = plateaus[-1] if which == "last" else plateaus[0]
    return int(a)
