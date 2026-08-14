#!/usr/bin/env python
"""Convert the raw datasets into the single layout the evaluation reads.

Reads what ``scripts/download_datasets.py`` put under ``datasets/raw/`` and
writes one folder per sample::

    datasets/formatted/<dataset>/<image_id>/image.jpg   # RGB image
    datasets/formatted/<dataset>/<image_id>/mask.jpg    # binary mask, 0 or 255

All three datasets ship genuine object/saliency masks, they just store them
differently:

  * ``cub``    - grayscale segmentation PNG, paired through the class/name path
  * ``ecssd``  - binary PNG mask, paired by image id (0001, 0002, ...)
  * ``msra_b`` - binary PNG mask, sharing the image's basename in one flat folder

Masks are thresholded at 127, so the ``.jpg`` container is lossless in practice
(verified bit-identical to the source PNGs after thresholding).

Samples that already exist are skipped, so re-running is cheap and interrupted
runs resume where they stopped. Use ``--overwrite`` to rebuild from scratch.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
DEFAULT_RAW = REPO / "datasets" / "raw"
DEFAULT_OUT = REPO / "datasets" / "formatted"

DATASETS = ("cub", "ecssd", "msra_b")


@dataclass(frozen=True)
class Job:
    """One image/mask pair to convert."""

    out_dir: Path
    image_path: Path
    mask_path: Path


def convert(job: Job) -> None:
    """Write ``image.jpg`` + binarized ``mask.jpg`` into ``job.out_dir``."""
    job.out_dir.mkdir(parents=True, exist_ok=True)
    Image.open(job.image_path).convert("RGB").save(job.out_dir / "image.jpg", quality=95)
    mask = np.asarray(Image.open(job.mask_path).convert("L"))
    binary = (mask > 127).astype(np.uint8) * 255
    Image.fromarray(binary, mode="L").save(job.out_dir / "mask.jpg", quality=95)


# --------------------------------------------------------------------------- #
# per-dataset pairing: raw layout -> list of jobs
# --------------------------------------------------------------------------- #
def jobs_cub(raw: Path, out: Path) -> list[Job]:
    """Masks drive the pairing: images.txt is not needed and the 8 stray
    ``*_rgb.jpg`` duplicates in the tarball have no mask, so they drop out."""
    images, segs = raw / "CUB_200_2011" / "images", raw / "segmentations"
    jobs = []
    for seg in sorted(segs.rglob("*.png")):
        image = images / seg.relative_to(segs).with_suffix(".jpg")  # <class>/<name>.jpg
        if image.exists():
            jobs.append(Job(out / "cub" / seg.stem, image, seg))
    return jobs


def jobs_ecssd(raw: Path, out: Path) -> list[Job]:
    images, masks = raw / "ECSSD" / "images", raw / "ECSSD" / "ground_truth_mask"
    jobs = []
    for image in sorted(images.glob("*.jpg")):
        mask = masks / f"{image.stem}.png"
        if mask.exists():
            jobs.append(Job(out / "ecssd" / image.stem, image, mask))
    return jobs


def jobs_msra_b(raw: Path, out: Path) -> list[Job]:
    folder = raw / "MSRA-B"
    jobs = []
    for image in sorted(folder.glob("*.jpg")):
        mask = image.with_suffix(".png")
        if mask.exists():
            jobs.append(Job(out / "msra_b" / image.stem, image, mask))
    return jobs


BUILDERS = {"cub": jobs_cub, "ecssd": jobs_ecssd, "msra_b": jobs_msra_b}

# Where each dataset's raw files must live for it to be convertible.
REQUIRED = {
    "cub": ("CUB_200_2011/images", "segmentations"),
    "ecssd": ("ECSSD/images", "ECSSD/ground_truth_mask"),
    "msra_b": ("MSRA-B",),
}


def done(job: Job) -> bool:
    return (job.out_dir / "image.jpg").exists() and (job.out_dir / "mask.jpg").exists()


def format_dataset(
    name: str, raw: Path, out: Path, limit: int | None, workers: int, overwrite: bool
) -> int:
    """Convert one dataset; returns the number of pairs available on disk after."""
    missing = [p for p in REQUIRED[name] if not (raw / p).is_dir()]
    if missing:
        print(f"[{name:6s}] SKIP — missing {', '.join(str(raw / p) for p in missing)}")
        print(f"[{name:6s}]        run: python scripts/download_datasets.py --datasets {name}")
        return 0

    jobs = BUILDERS[name](raw, out)
    if limit:
        jobs = jobs[:limit]
    total = len(jobs)

    todo = jobs if overwrite else [j for j in jobs if not done(j)]
    if not todo:
        print(f"[{name:6s}] {total} pairs already formatted — nothing to do")
        return total

    print(f"[{name:6s}] {len(todo)} to convert ({total - len(todo)} already done)")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for _ in tqdm(
            pool.map(convert, todo, chunksize=32),
            total=len(todo), desc=f"[{name:6s}]", unit="pair",
        ):
            pass
    return total


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=DEFAULT_RAW,
        help=f"where the downloaded datasets live (default: {DEFAULT_RAW.relative_to(REPO)})",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUT,
        help=f"where to write the unified layout (default: {DEFAULT_OUT.relative_to(REPO)})",
    )
    parser.add_argument(
        "--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS),
        help="which datasets to convert (default: all three)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="convert at most N samples per dataset (quick test run)",
    )
    parser.add_argument(
        "--workers", type=int, default=8, help="parallel conversion processes (default: 8)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="re-convert samples that are already formatted",
    )
    args = parser.parse_args()

    print(f"Reading  {args.raw_dir}")
    print(f"Writing  {args.output}")
    counts = {
        name: format_dataset(
            name, args.raw_dir, args.output, args.limit, args.workers, args.overwrite
        )
        for name in args.datasets
    }

    print("\nFormatted datasets:")
    for name, count in counts.items():
        print(f"  {name:6s} {count:>6d} pairs")
    total = sum(counts.values())
    print(f"  {'total':6s} {total:>6d} pairs under {args.output}")
    if total:
        print("\nNext: python -m experiments.foreground_extraction.eval")
    return 0 if all(counts.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
