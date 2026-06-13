#!/usr/bin/env python
"""Turn a multi-panel "contour evolution" strip (as produced by
``deep_contourflow.visualization.plot_contour_evolution``) into an animated GIF.

The strip is split back into its individual step panels by detecting the
cream-coloured background gaps between them, then the panels are assembled into
a looping GIF.

Usage
-----
    python scripts/make_evolution_gif.py assets/lion.png assets/contour_evolution.gif
"""
import argparse

import numpy as np
from PIL import Image

CREAM = np.array([250, 249, 246])


def detect_panels(arr, bg_tol=25, gap_frac=0.97, min_width=40):
    """Return (start, end) x-ranges of the panels in a strip image."""
    is_bg = np.abs(arr - CREAM).sum(axis=2) < bg_tol
    gap = is_bg.mean(axis=0) > gap_frac
    panels, start = [], None
    for x in range(arr.shape[1]):
        if not gap[x] and start is None:
            start = x
        elif gap[x] and start is not None:
            if x - start > min_width:
                panels.append((start, x))
            start = None
    if start is not None and arr.shape[1] - start > min_width:
        panels.append((start, arr.shape[1]))
    return panels


def make_gif(src, dst, duration=700, end_pause=1600, pad=8):
    im = Image.open(src).convert("RGB")
    arr = np.asarray(im).astype(int)
    panels = detect_panels(arr)
    if len(panels) < 2:
        raise SystemExit(f"Only {len(panels)} panel(s) detected in {src}; nothing to animate.")

    w = max(e - s for s, e in panels)
    h = im.height
    frames = []
    for s, e in panels:
        canvas = Image.new("RGB", (w + 2 * pad, h + 2 * pad), tuple(CREAM))
        panel = im.crop((s, 0, e, h))
        canvas.paste(panel, (pad + (w - (e - s)) // 2, pad))
        frames.append(canvas)

    # Linger on the final (converged) frame.
    durations = [duration] * (len(frames) - 1) + [end_pause]
    frames[0].save(
        dst, save_all=True, append_images=frames[1:],
        duration=durations, loop=0, optimize=True,
    )
    print(f"Wrote {dst} — {len(frames)} frames, {w + 2 * pad}x{h + 2 * pad}px")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("src", help="input strip PNG")
    p.add_argument("dst", help="output GIF path")
    p.add_argument("--duration", type=int, default=700, help="ms per frame")
    p.add_argument("--end-pause", type=int, default=1600, help="ms on final frame")
    args = p.parse_args()
    make_gif(args.src, args.dst, args.duration, args.end_pause)
