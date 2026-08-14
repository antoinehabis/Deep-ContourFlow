"""DCF foreground-extraction experiment package.

Modules
-------
dataset  - ForegroundDataset, dataloader, frame crop/remap and contour helpers.
metrics  - segmentation metrics + SegmentationMetrics accumulator.
eval     - Hydra entry point + ``evaluate`` (python -m experiments.foreground_extraction.eval).
"""

from typing import Any

from .metrics import SegmentationMetrics, compute_all

__all__ = ["evaluate", "load_cfg", "SegmentationMetrics", "compute_all"]


def __getattr__(name: str) -> Any:
    """Expose ``evaluate``/``load_cfg`` lazily.

    Importing ``.eval`` eagerly would pull the module in before
    ``python -m experiments.foreground_extraction.eval`` executes it, which makes
    runpy warn about a double import.
    """
    if name in ("evaluate", "load_cfg"):
        from . import eval as _eval

        return getattr(_eval, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
