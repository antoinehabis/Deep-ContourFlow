"""
DCF visualisation utilities.

Functions
---------
plot_support_with_gt(img, gt, ...)
    Two-panel figure: raw support image (left) + image with GT contour overlay (right).

plot_contour_evolution(img, contour_history, losses, ...)
    n_steps evenly-spaced contour snapshots laid out horizontally.

Usage
-----
    from script_visualisation.visualize_contour_evolution import (
        plot_support_with_gt,
        plot_contour_evolution,
    )

    fig = plot_support_with_gt(img, gt, filename="human1.jpg")

    contours, scores, losses, energies = dcf.predict(tensor_query, contour_init)
    fig = plot_contour_evolution(img_query, dcf.contour_history_, losses)

batch_idx (plot_contour_evolution)
-----------------------------------
- int  → show that single element only  (1 row × n_steps cols)
- None → show every element in the batch (B rows × n_steps cols)

img shapes accepted
-------------------
- (H, W, 3)    — same image broadcast to all rows / panels
- (B, H, W, 3) — one image per batch element
"""

from __future__ import annotations

import os
import urllib.request
import matplotlib.font_manager as _fm
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def _ensure_poppins() -> bool:
    """Download Poppins from Google Fonts if not already registered. Returns True on success."""
    if any(f.name == "Poppins" for f in _fm.fontManager.ttflist):
        return True

    font_dir = os.path.join(os.path.expanduser("~"), ".cache", "matplotlib", "fonts")
    os.makedirs(font_dir, exist_ok=True)

    _POPPINS_URLS = {
        "Poppins-Light.ttf":    "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Light.ttf",
        "Poppins-Regular.ttf":  "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Regular.ttf",
        "Poppins-Medium.ttf":   "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-Medium.ttf",
        "Poppins-SemiBold.ttf": "https://github.com/google/fonts/raw/main/ofl/poppins/Poppins-SemiBold.ttf",
    }

    try:
        for fname, url in _POPPINS_URLS.items():
            dest = os.path.join(font_dir, fname)
            if not os.path.exists(dest):
                print(f"Downloading {fname} …")
                urllib.request.urlretrieve(url, dest)
            _fm.fontManager.addfont(dest)
        return True
    except Exception as exc:
        print(f"[visualize_contour_evolution] Could not download Poppins ({exc}) — using DejaVu Sans")
        return False


if _ensure_poppins():
    plt.rcParams["font.family"] = "Poppins"

# ── Design tokens ─────────────────────────────────────────────────────────────
BG        = "#FAF9F6"
GRAY_DARK = "#3A3A3A"
GRAY_MID  = "#7A7A7A"

# Temporal gradient: warm (early) → cool (late)
_STEP_COLORS = ["#F97B6B", "#F5C842", "#3ECFB2", "#6BCFF5", "#5B6CF9"]
# ──────────────────────────────────────────────────────────────────────────────


def _to_uint8(img) -> np.ndarray:
    # Accept numpy arrays, torch tensors, or anything array-like
    try:
        arr = img.detach().cpu().numpy() if hasattr(img, "detach") else np.asarray(img)
    except Exception:
        arr = np.asarray(img)

    # Drop leading batch dim: (1, ...) → (...)
    if arr.ndim == 4:
        arr = arr[0]

    # Channels-first (C, H, W) → channels-last (H, W, C)
    if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[2] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)

    # Grayscale (H, W, 1) → (H, W, 3)
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)

    if arr.dtype == np.uint8:
        return arr

    # Float: detect range then scale
    vmax = float(arr.max())
    if vmax > 1.0:
        return np.clip(arr, 0, 255).astype(np.uint8)
    return (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)


def _draw_row(
    axes: np.ndarray,
    img: np.ndarray,
    contour_history: np.ndarray,
    losses: np.ndarray,
    b: int,
    step_indices: np.ndarray,
    colors: list,
    row_label: Optional[str] = None,
) -> None:
    """Fill one row of axes for batch element `b`."""
    for col, (epoch_idx, color) in enumerate(zip(step_indices, colors)):
        ax = axes[col]
        contour_xy = contour_history[epoch_idx, b].reshape(-1, 2)
        loss_val = float(losses[epoch_idx, b])

        # Show image and lock limits immediately so the contour overlay
        # cannot trigger matplotlib's autoscale and push the image away.
        ax.imshow(img)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        # Contour as a closed line + semi-transparent fill
        closed = np.vstack([contour_xy, contour_xy[:1]])
        ax.fill(contour_xy[:, 0], contour_xy[:, 1],
                color=color, alpha=0.18, zorder=4)
        ax.plot(closed[:, 0], closed[:, 1],
                color=color, linewidth=2.5, zorder=5, solid_capstyle="round")

        # Restore image limits
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.axis("off")

        if row_label is None or b == 0:
            ax.set_title(
                f"step {epoch_idx + 1}",
                fontsize=10, fontweight=500, color=GRAY_DARK, pad=8,
            )
        ax.text(
            0.5, -0.10,
            f"loss: {loss_val:.4f}",
            transform=ax.transAxes, ha="center",
            fontsize=8.5, fontweight=300, color=GRAY_MID,
        )

    if row_label is not None:
        axes[0].set_ylabel(
            row_label, fontsize=9, color=GRAY_MID, rotation=90, labelpad=6,
        )
        axes[0].yaxis.set_visible(True)


def plot_support_with_gt(
    img: np.ndarray,
    gt: np.ndarray,
    filename: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Two-panel figure: raw support image (left) and image + GT contour overlay (right).

    Parameters
    ----------
    img       : (H, W, 3) uint8 or float [0, 1]
    gt        : (H, W) or (H, W, 1) uint8 binary mask
    filename  : optional label shown as subtitle under the left panel title
    save_path : path to save the figure (optional)
    """
    img_u8 = _to_uint8(img)

    gt_arr = np.asarray(gt, dtype=np.float32)
    if gt_arr.ndim == 3:
        gt_arr = gt_arr[..., 0]
    gt_arr = (gt_arr > 0).astype(np.float32)

    GT_COLOR = "#5B6CF9"  # indigo

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.5))
    fig.patch.set_facecolor(BG)

    ax.imshow(img_u8)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()

    ax.contourf(gt_arr, levels=[0.5, 1.5], colors=[GT_COLOR], alpha=0.20, zorder=4)
    ax.contour(gt_arr, levels=[0.5], colors=[GT_COLOR], linewidths=2.2, zorder=5)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axis("off")
    ax.set_title(
        "Ground truth contour",
        fontsize=11, fontweight=500, color=GRAY_DARK, pad=10, loc="left",
    )
    if filename:
        ax.text(
            0.98, 0.02, filename,
            transform=ax.transAxes,
            fontsize=8.5, fontstyle="italic", fontweight=300, color=GRAY_MID,
            va="bottom", ha="right",
        )

    plt.tight_layout(pad=0.4)

    if save_path is not None:
        fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=BG)
        print(f"Saved → {save_path}")

    return fig


def plot_contour_evolution(
    img: np.ndarray,
    contour_history: np.ndarray,
    losses: np.ndarray,
    batch_idx: Optional[int] = 0,
    n_steps: int = 5,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Parameters
    ----------
    img              : (H, W, 3) or (B, H, W, 3) — uint8 or float [0, 1]
    contour_history  : (n_epochs, B, ..., K, 2) in [x, y] pixel coords
    losses           : (n_epochs, B) or (B, n_epochs) — orientation auto-detected
    batch_idx        : int → single element; None → all batch elements
    n_steps          : number of evenly-spaced snapshots (default 5)
    save_path        : path to save the figure (optional)

    Returns
    -------
    matplotlib.figure.Figure
    """
    contour_history = np.asarray(contour_history)
    n_valid = contour_history.shape[0]
    B = contour_history.shape[1]

    losses = np.asarray(losses)
    if losses.ndim == 2 and losses.shape[0] != n_valid:
        losses = losses.T

    step_indices = np.linspace(0, n_valid - 1, n_steps, dtype=int)
    colors = _STEP_COLORS[:n_steps]

    batch_elements = [batch_idx] if isinstance(batch_idx, int) else list(range(B))
    n_rows = len(batch_elements)

    img_arr = np.asarray(img)
    if img_arr.ndim == 3:
        imgs = [_to_uint8(img_arr)] * n_rows
    else:
        imgs = [_to_uint8(img_arr[b]) for b in batch_elements]

    fig, axes_grid = plt.subplots(
        n_rows, n_steps,
        figsize=(3.6 * n_steps, 4.5 * n_rows),
        gridspec_kw={"wspace": 0.05, "hspace": 0.28},
        squeeze=False,
    )
    fig.patch.set_facecolor(BG)

    fig.text(
        0.015, 1.0 if n_rows == 1 else 1.01,
        "Contour Evolution",
        fontsize=14, fontweight=500, color=GRAY_DARK,
        va="bottom", ha="left",
    )

    for row, b in enumerate(batch_elements):
        row_label = f"sample {b}" if n_rows > 1 else None
        _draw_row(
            axes_grid[row], imgs[row], contour_history, losses,
            b, step_indices, colors, row_label,
        )

    plt.tight_layout(pad=0.4)

    if save_path is not None:
        fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=BG)
        print(f"Saved → {save_path}")

    return fig
