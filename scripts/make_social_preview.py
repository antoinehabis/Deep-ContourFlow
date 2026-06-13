#!/usr/bin/env python
"""Generate the GitHub social-preview banner (1280x640) for Deep ContourFlow.

Composites the converged contour result next to the project title/tagline.

Usage
-----
    python scripts/make_social_preview.py assets/lion.png assets/social_preview.png
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

CREAM = "#FAF9F6"
INK = "#2E2E2E"
MUTED = "#8A8A8A"
CORAL = "#E0796F"


def last_panel(src):
    """Crop the right-most (converged) panel from an evolution strip."""
    im = Image.open(src).convert("RGB")
    arr = np.abs(np.asarray(im).astype(int) - np.array([250, 249, 246])).sum(2) < 25
    gap = arr.mean(0) > 0.97
    runs, start = [], None
    for x in range(im.width):
        if not gap[x] and start is None:
            start = x
        elif gap[x] and start is not None:
            if x - start > 40:
                runs.append((start, x))
            start = None
    if start is not None:
        runs.append((start, im.width))
    s, e = runs[-1]
    return im.crop((s, 0, e, im.height))


def make_banner(src, dst):
    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)
    fig.patch.set_facecolor(CREAM)

    # Right: converged result
    ax = fig.add_axes([0.60, 0.08, 0.36, 0.84])
    ax.imshow(last_panel(src))
    ax.axis("off")

    # Left: title block
    fig.text(0.06, 0.66, "Deep ContourFlow", fontsize=46, fontweight="bold", color=INK)
    fig.text(0.063, 0.55, "Training-free active contours,\npowered by deep features",
             fontsize=22, color=INK, linespacing=1.3)
    fig.text(0.065, 0.30, "Unsupervised  ·  One-shot  ·  PyTorch", fontsize=16, color=CORAL,
             fontweight="bold")
    fig.text(0.065, 0.22, "No training · no labels · just a frozen CNN", fontsize=14, color=MUTED)
    fig.text(0.065, 0.13, "arXiv:2407.10696", fontsize=13, color=MUTED, style="italic")

    fig.savefig(dst, facecolor=CREAM)
    plt.close(fig)
    print(f"Wrote {dst}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src", help="evolution strip PNG (uses its converged panel)")
    p.add_argument("dst", help="output banner path")
    args = p.parse_args()
    make_banner(args.src, args.dst)
