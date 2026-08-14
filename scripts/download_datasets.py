#!/usr/bin/env python
"""Download the raw benchmark datasets (CUB-200-2011, ECSSD, MSRA-B).

Each archive is fetched from its official host into a cache directory, checked
against a known size and checksum, and expanded under ``datasets/raw/``::

    datasets/raw/CUB_200_2011/images/<class>/<image>.jpg
    datasets/raw/segmentations/<class>/<image>.png
    datasets/raw/ECSSD/images/<id>.jpg
    datasets/raw/ECSSD/ground_truth_mask/<id>.png
    datasets/raw/MSRA-B/<id>.jpg + <id>.png

Downloads resume where they stopped, and a dataset whose files are already on
disk is skipped, so the script is safe to re-run. Nothing is ever deleted unless
``--delete-archives`` is passed explicitly.

Next step: ``python scripts/format_datasets.py`` turns this into the unified
layout the evaluation reads.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

REPO = Path(__file__).resolve().parent.parent
DEFAULT_RAW = REPO / "datasets" / "raw"
DEFAULT_CACHE = REPO / "datasets" / "archives"

# Some academic hosts reject the default urllib user agent.
USER_AGENT = "Mozilla/5.0 (compatible; deep-contourflow-dataset-downloader)"
CHUNK = 1 << 20  # 1 MiB


@dataclass(frozen=True)
class Archive:
    """One downloadable archive and where its contents belong."""

    url: str
    size: int  # exact content-length, verified after download
    checksum: str  # "<algo>:<hexdigest>", e.g. "md5:97ecee..." or "sha256:7d5dce..."
    extract_into: str  # destination dir, relative to the raw root
    produces: str  # path (relative to raw root) that exists once extracted

    @property
    def filename(self) -> str:
        return self.url.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class Dataset:
    """A benchmark dataset: its archives, provenance and an integrity check."""

    key: str
    title: str
    homepage: str
    archives: tuple[Archive, ...]
    # (path relative to raw root, glob, expected number of matches)
    expected: tuple[tuple[str, str, int], ...] = field(default=())


DATASETS: dict[str, Dataset] = {
    "cub": Dataset(
        key="cub",
        title="CUB-200-2011 (11 788 birds, with segmentation masks)",
        homepage="https://www.vision.caltech.edu/datasets/cub_200_2011/",
        archives=(
            Archive(
                url="https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz",
                size=1150585339,
                checksum="md5:97eceeb196236b17998738112f37df78",
                extract_into=".",
                produces="CUB_200_2011/images",
            ),
            Archive(
                url="https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz",
                size=39272883,
                checksum="md5:4d47ba1228eae64f2fa547c47bc65255",
                extract_into=".",
                produces="segmentations",
            ),
        ),
        expected=(
            # 11 788 indexed images + 8 stray "*_rgb.jpg" duplicates of grayscale
            # originals that ship in the tarball but are absent from images.txt.
            # They have no segmentation mask, so the formatter never picks them up.
            ("CUB_200_2011/images", "**/*.jpg", 11796),
            ("segmentations", "**/*.png", 11788),
        ),
    ),
    "ecssd": Dataset(
        key="ecssd",
        title="ECSSD (1 000 complex-scene saliency images)",
        homepage="https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html",
        archives=(
            Archive(
                url="https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/data/ECSSD/images.zip",
                size=67766979,
                checksum="sha256:7d5dce1a21c6d82e2f9617c00ad7eae2de5ded9320e8c59a01b4c731d210ea47",
                extract_into="ECSSD",
                produces="ECSSD/images",
            ),
            Archive(
                url="https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/data/ECSSD/ground_truth_mask.zip",
                size=1571109,
                checksum="sha256:dc0a9f11f2adee2737d95b28f9567a0a5de33f78743447c097ceb4d42f281a31",
                extract_into="ECSSD",
                produces="ECSSD/ground_truth_mask",
            ),
        ),
        expected=(
            ("ECSSD/images", "*.jpg", 1000),
            ("ECSSD/ground_truth_mask", "*.png", 1000),
        ),
    ),
    "msra_b": Dataset(
        key="msra_b",
        title="MSRA-B (5 000 salient-object images)",
        homepage="https://mmcheng.net/msra10k/",
        archives=(
            Archive(
                url="https://mftp.mmcheng.net/Data/MSRA-B.zip",
                size=114133760,
                checksum="sha256:1c82ee25d81284922c1d3bd7be971d1010999bb7a85b6fefc5d3246ee07937d8",
                extract_into=".",  # the zip already carries a top-level MSRA-B/ folder
                produces="MSRA-B",
            ),
        ),
        expected=(
            ("MSRA-B", "*.jpg", 5000),
            ("MSRA-B", "*.png", 5000),
        ),
    ),
}


# --------------------------------------------------------------------------- #
# download / verify / extract
# --------------------------------------------------------------------------- #
def human_size(num: float) -> str:
    for unit in ("B", "KiB", "MiB"):
        if num < 1024:
            return f"{num:.0f} {unit}"
        num /= 1024
    return f"{num:.2f} GiB"


def file_digest(path: Path, algo: str) -> str:
    """Stream-hash a file so multi-GB archives never land in memory."""
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def check_archive(path: Path, archive: Archive) -> bool:
    """True when ``path`` matches the archive's expected size *and* checksum."""
    if not path.is_file() or path.stat().st_size != archive.size:
        return False
    algo, expected = archive.checksum.split(":", 1)
    return file_digest(path, algo) == expected


def download(archive: Archive, dest: Path) -> None:
    """Fetch ``archive`` to ``dest``, resuming a partial ``.part`` file if present.

    Writes to ``dest.part`` and renames only once the checksum matches, so an
    interrupted run can never leave a corrupt archive behind under ``dest``.
    """
    part = dest.with_suffix(dest.suffix + ".part")
    have = part.stat().st_size if part.is_file() else 0
    if have > archive.size:  # stale/foreign leftover -> start over
        part.unlink()
        have = 0

    if have < archive.size:
        request = urllib.request.Request(archive.url, headers={"User-Agent": USER_AGENT})
        if have:
            request.add_header("Range", f"bytes={have}-")
        try:
            response = urllib.request.urlopen(request, timeout=60)
        except urllib.error.HTTPError as exc:
            if have and exc.code in (416, 501):  # server refuses ranges -> restart
                part.unlink()
                have = 0
                request = urllib.request.Request(
                    archive.url, headers={"User-Agent": USER_AGENT}
                )
                response = urllib.request.urlopen(request, timeout=60)
            else:
                raise
        # A 200 to a Range request means the server ignored it and restarted.
        if have and response.status == 200:
            have = 0

        mode = "ab" if have else "wb"
        with response, open(part, mode) as out, tqdm(
            total=archive.size, initial=have, unit="B", unit_scale=True,
            unit_divisor=1024, desc=f"  {archive.filename}",
        ) as bar:
            for block in iter(lambda: response.read(CHUNK), b""):
                out.write(block)
                bar.update(len(block))

    actual = part.stat().st_size
    if actual != archive.size:
        raise RuntimeError(
            f"{archive.filename}: got {actual} bytes, expected {archive.size}. "
            f"Delete {part} and re-run."
        )
    algo, expected = archive.checksum.split(":", 1)
    print(f"  verifying {algo} ...", end=" ", flush=True)
    digest = file_digest(part, algo)
    if digest != expected:
        raise RuntimeError(
            f"{archive.filename}: {algo} mismatch\n  expected {expected}\n  got      {digest}\n"
            f"The download is corrupt or the host changed the file. Delete {part} and re-run."
        )
    print("ok")
    part.rename(dest)


def _is_within(base: Path, target: Path) -> bool:
    """Guard against archive members escaping the destination (zip-slip)."""
    return base.resolve() in target.resolve().parents or base.resolve() == target.resolve()


def extract(path: Path, dest: Path) -> None:
    """Expand a .zip or .tar.* archive into ``dest``, rejecting unsafe members."""
    dest.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            members = zf.namelist()
            for name in members:
                if not _is_within(dest, dest / name):
                    raise RuntimeError(f"{path.name}: unsafe member path {name!r}")
            for name in tqdm(members, desc=f"  extracting {path.name}", unit="file"):
                zf.extract(name, dest)
    else:
        with tarfile.open(path) as tf:
            members = tf.getmembers()
            for member in members:
                if not _is_within(dest, dest / member.name):
                    raise RuntimeError(f"{path.name}: unsafe member path {member.name!r}")
            for member in tqdm(members, desc=f"  extracting {path.name}", unit="file"):
                tf.extract(member, dest)


# --------------------------------------------------------------------------- #
# per-dataset driver
# --------------------------------------------------------------------------- #
def verify(dataset: Dataset, raw_dir: Path) -> bool:
    """Report the file counts found on disk against what the dataset should have."""
    ok = True
    for subpath, pattern, count in dataset.expected:
        found = len(list((raw_dir / subpath).glob(pattern))) if (raw_dir / subpath).is_dir() else 0
        status = "ok " if found == count else "MISMATCH"
        print(f"  [{status}] {subpath}/{pattern}: {found}/{count}")
        ok &= found == count
    return ok


def fetch(dataset: Dataset, raw_dir: Path, cache_dir: Path, force: bool) -> None:
    """Download + extract every archive of ``dataset`` that is not already in place."""
    print(f"\n=== {dataset.key} — {dataset.title}")
    print(f"    {dataset.homepage}")
    for archive in dataset.archives:
        target = raw_dir / archive.produces
        if target.exists() and not force:
            print(f"  {archive.produces} already present — skipping")
            continue

        cached = cache_dir / archive.filename
        if check_archive(cached, archive):
            print(f"  using cached {archive.filename}")
        else:
            print(f"  downloading {archive.filename} ({human_size(archive.size)})")
            download(archive, cached)
        extract(cached, raw_dir / archive.extract_into)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS),
        help="which datasets to download (default: all three)",
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=DEFAULT_RAW,
        help=f"where to extract the datasets (default: {DEFAULT_RAW.relative_to(REPO)})",
    )
    parser.add_argument(
        "--cache-dir", type=Path, default=DEFAULT_CACHE,
        help=f"where to keep the downloaded archives (default: {DEFAULT_CACHE.relative_to(REPO)})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="re-extract even when the target directory already exists",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="only check what is already on disk; download nothing",
    )
    parser.add_argument(
        "--delete-archives", action="store_true",
        help="remove the downloaded archives once extracted (frees ~1.3 GiB)",
    )
    args = parser.parse_args()

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for key in args.datasets:
        dataset = DATASETS[key]
        if not args.verify_only:
            try:
                fetch(dataset, args.raw_dir, args.cache_dir, args.force)
            except (OSError, RuntimeError, urllib.error.URLError) as exc:
                print(f"\n  FAILED ({key}): {exc}", file=sys.stderr)
                print(f"  Manual download: {dataset.homepage}", file=sys.stderr)
                all_ok = False
                continue
        else:
            print(f"\n=== {dataset.key} — {dataset.title}")
        all_ok &= verify(dataset, args.raw_dir)

    if args.delete_archives and all_ok:
        # Only ever remove archives this script downloaded — the cache directory
        # may hold unrelated files that are none of our business.
        for key in args.datasets:
            for archive in DATASETS[key].archives:
                cached = args.cache_dir / archive.filename
                if cached.is_file():
                    print(f"Removing {cached}")
                    cached.unlink()

    print(
        f"\n{'All datasets ready' if all_ok else 'Some datasets are incomplete'} "
        f"in {args.raw_dir}"
    )
    if all_ok:
        print("Next: python scripts/format_datasets.py")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
