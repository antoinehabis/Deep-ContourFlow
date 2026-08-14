"""Evaluate unsupervised DCF as a foreground extractor. Hydra entry point.

Config lives in ``conf/config.yaml``; override any field with Hydra's
``key=value`` syntax. Run from the repo root.

Examples
--------
# Full benchmark on all three datasets (the published numbers)
python -m experiments.foreground_extraction.eval

# Quick check: 200 seeded images per dataset (the numbers in the README)
python -m experiments.foreground_extraction.eval data.max_samples=200

# One dataset, a few images
python -m experiments.foreground_extraction.eval 'datasets=[ecssd]' data.max_samples=8

# Sweep a hyperparameter (-m runs each value in its own output dir)
python -m experiments.foreground_extraction.eval -m dcf.learning_rate=0.01,0.05

Programmatic use
----------------
>>> from experiments.foreground_extraction.eval import load_cfg, evaluate
>>> evaluate(load_cfg(), "ecssd")["metrics"]
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from deep_contourflow import UnsupervisedDCF

from .dataset import (
    build_dataloader,
    contours_to_masks,
    make_init_contours,
    remap_masks_to_full,
)
from .metrics import METRICS, SegmentationMetrics

REPO = Path(__file__).resolve().parents[2]
CONF = Path(__file__).resolve().parent / "conf" / "config.yaml"
_SHORT = {"iou": "iou", "dice": "dice", "pixel_accuracy": "pacc"}  # tqdm postfix labels


def load_cfg(path: str | Path = CONF) -> DictConfig:
    """Load ``conf/config.yaml`` for programmatic use (notebooks, scripts)."""
    cfg = OmegaConf.load(path)
    assert isinstance(cfg, DictConfig)
    return cfg


def _resolve(path: str | Path) -> Path:
    """Resolve a (possibly relative) path against the repo root."""
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def _pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _predict_masks(dcf: UnsupervisedDCF, images: torch.Tensor, cfg: DictConfig, device: str):
    """Run DCF on a batch ``(B, C, H, W)`` -> content-space binary masks ``(B, H, W)``.

    No ``torch.no_grad``: ``DCF.predict`` runs its own autograd contour optimization.
    """
    image_size = images.shape[-1]
    images = images.to(torch.float32).to(device)
    contour_init = make_init_contours(
        images.shape[0], image_size,
        cfg.init.shape, cfg.init.size, cfg.init.nb_nodes, device,
    )
    _, _, final_contours = dcf.predict(images, contour_init)
    return contours_to_masks(final_contours, image_size)


def evaluate(cfg: DictConfig, dataset_name: str, dcf: UnsupervisedDCF | None = None) -> dict:
    """Run DCF over one dataset and return aggregated metrics.

    ``dcf`` lets a caller share one initialized model across datasets; when it is
    ``None`` a fresh ``UnsupervisedDCF`` is built from ``cfg.dcf``.
    """
    device = _pick_device(cfg.device)
    root = _resolve(cfg.data.root) / dataset_name
    loader = build_dataloader(
        root=root,
        image_size=cfg.data.image_size,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        max_samples=cfg.data.max_samples,
        shuffle=cfg.data.shuffle,
        remove_frame=cfg.data.remove_frame,
        seed=cfg.data.seed,
    )
    # Hide DCF's per-step logging; its inner "Optimizing contour" bar is hidden by
    # redirecting stderr only around the predict call below.
    logging.getLogger("deep_contourflow").setLevel(logging.WARNING)
    if dcf is None:
        dcf = UnsupervisedDCF(**OmegaConf.to_container(cfg.dcf, resolve=True))
    meter = SegmentationMetrics()

    start = time.perf_counter()
    pbar = tqdm(loader, desc=f"DCF/{dataset_name}", unit="batch")
    for batch in pbar:
        with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
            content_masks = _predict_masks(dcf, batch["image"], cfg, device)
        # Undo the frame crop so predictions align with the full-image ground truth.
        preds = remap_masks_to_full(content_masks, batch["crop_box"].numpy(), cfg.data.image_size)
        meter.update(preds, batch["mask"].numpy())
        pbar.set_postfix({_SHORT[k]: round(v, 3) for k, v in meter.average().items()})

    seconds = time.perf_counter() - start
    return {
        "dataset": dataset_name,
        "n_samples": len(meter),
        "device": device,
        "seconds": round(seconds, 2),
        "seconds_per_image": round(seconds / max(len(meter), 1), 3),
        "metrics": meter.average(),
    }


def macro_average(results: list[dict]) -> dict:
    """Mean of the per-dataset scores, weighting every dataset equally.

    Equal weight per dataset (not per image) so the 11 788-image CUB does not
    drown out the 1 000-image ECSSD.
    """
    return {
        name: sum(r["metrics"][name] for r in results) / len(results)
        for name in METRICS
    }


def _print_summary(results: list[dict], macro: dict) -> None:
    head = f"{'dataset':<10}{'n':>7}{'IoU':>9}{'Dice':>9}{'pixel acc':>11}{'s/img':>9}"
    print("\n" + head)
    print("-" * len(head))
    for r in results:
        m = r["metrics"]
        print(
            f"{r['dataset']:<10}{r['n_samples']:>7}{m['iou']:>9.4f}{m['dice']:>9.4f}"
            f"{m['pixel_accuracy']:>11.4f}{r['seconds_per_image']:>9.2f}"
        )
    print("-" * len(head))
    print(
        f"{'MACRO':<10}{'':>7}{macro['iou']:>9.4f}{macro['dice']:>9.4f}"
        f"{macro['pixel_accuracy']:>11.4f}"
    )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    missing = [
        name for name in cfg.datasets if not (_resolve(cfg.data.root) / name).is_dir()
    ]
    if missing:
        raise SystemExit(
            f"Missing formatted dataset(s): {', '.join(missing)} under {_resolve(cfg.data.root)}\n"
            "Run:  python scripts/download_datasets.py && python scripts/format_datasets.py"
        )

    # One model for every dataset: loading the backbone is the only shared setup cost.
    dcf = UnsupervisedDCF(**OmegaConf.to_container(cfg.dcf, resolve=True))
    results = [evaluate(cfg, name, dcf=dcf) for name in cfg.datasets]
    macro = macro_average(results)
    _print_summary(results, macro)

    payload = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "results": results,
        "macro_average": macro,
    }
    # Hydra gives each run its own timestamped directory; keep results.json beside
    # that run's logs, and refresh results/latest.json for convenience.
    run_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "results.json").write_text(json.dumps(payload, indent=2))

    latest = _resolve(cfg.output_dir) / "latest.json"
    latest.parent.mkdir(parents=True, exist_ok=True)
    latest.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved {run_dir / 'results.json'}\n      {latest}")


if __name__ == "__main__":
    main()
