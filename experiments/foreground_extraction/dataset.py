"""Dataset, dataloader and contour helpers for DCF foreground extraction.

The datasets are expected in the unified layout produced by
``scripts/format_datasets.py``::

    <root>/<image_id>/image.jpg
    <root>/<image_id>/mask.jpg

``ForegroundDataset`` yields, per sample, a float32 image tensor ``(C, H, W)`` in
[0, 1], a binary ground-truth mask ``(H, W)``, the ``image_id`` and a ``crop_box``
``(top, bottom, left, right)``. Everything lives in a square ``image_size`` space
so DCF's square-image assumption holds and metrics are consistent.

When ``remove_frame`` is on, a solid border frame is detected and cropped *before*
DCF sees the image: the cropped content is resized to ``image_size`` and fed to
DCF, and the predicted mask is later remapped back into the full ``image_size``
canvas (see ``remap_masks_to_full``) so it aligns with the ground truth. This
stops DCF from locking onto the frame edge instead of the object inside it.

Helpers:
  * ``detect_frame_crop``     - find a solid border frame to crop (conservative).
  * ``build_dataloader``      - a torch DataLoader over one dataset root.
  * ``make_init_contours``    - the initial circle/square contour, batched.
  * ``contours_to_masks``     - rasterize DCF's final contours to binary masks.
  * ``remap_masks_to_full``   - place cropped-space masks back into the full canvas.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch_contour import CleanContours

from deep_contourflow.features import define_contour_init


def detect_frame_crop(
    img: np.ndarray,
    line_std_max: float = 6.0,
    color_tol: float = 12.0,
    jump_min: float = 45.0,
    max_frac: float = 0.25,
    min_thick: int = 3,
) -> tuple[int, int, int, int]:
    """Detect a solid border frame and return its inner box ``(top, bottom, left, right)``.

    Conservative on purpose: a side is only cropped when its border is a near-solid
    color (``line_std_max``) consistent inward (``color_tol``) *and* gives way to the
    content through a sharp transition (``jump_min``). Smooth backgrounds (sky, grass,
    studio backdrops) drift gradually and are therefore left untouched. Returns the
    full-image box (no crop) when no frame is found.

    ``img`` is ``(H, W, 3)`` uint8.
    """
    h, w = img.shape[:2]
    f = img.astype(np.float32)

    def side(get_line, limit: int) -> int:
        ref = get_line(0).reshape(-1, 3).mean(0)
        i = 0
        while i < limit:
            line = get_line(i).reshape(-1, 3)
            if line.std() < line_std_max and np.abs(line.mean(0) - ref).mean() < color_tol:
                i += 1
            else:
                break
        if i < min_thick:
            return 0
        content = get_line(i).reshape(-1, 3).mean(0)  # first line past the frame
        return i if np.abs(content - ref).mean() > jump_min else 0

    top = side(lambda i: f[i, :, :], int(h * max_frac))
    bottom = side(lambda i: f[h - 1 - i, :, :], int(h * max_frac))
    left = side(lambda i: f[:, i, :], int(w * max_frac))
    right = side(lambda i: f[:, w - 1 - i, :], int(w * max_frac))

    t, b, l, r = top, h - bottom, left, w - right
    if b - t < 0.4 * h or r - l < 0.4 * w:  # degenerate -> trust nothing, keep full image
        return 0, h, 0, w
    return t, b, l, r


class ForegroundDataset(Dataset):
    """Image + binary mask pairs from a ``formatted_datasets`` root."""

    def __init__(
        self,
        root: Path,
        image_size: int = 512,
        max_samples: Optional[int] = None,
        remove_frame: bool = True,
        seed: Optional[int] = 1234,
    ):
        self.root = Path(root)
        self.image_size = image_size
        self.remove_frame = remove_frame
        if not self.root.is_dir():
            raise FileNotFoundError(
                f"Dataset root not found: {self.root}\n"
                "Run: python scripts/download_datasets.py && python scripts/format_datasets.py"
            )
        ids = sorted(p.name for p in self.root.iterdir() if (p / "image.jpg").exists())
        # Subsample with a fixed seed rather than truncating: the ids are sorted by
        # class/name, so ids[:N] would draw a single class on CUB. Re-sorted after
        # sampling so the run order stays deterministic.
        if max_samples is not None and max_samples < len(ids):
            ids = sorted(random.Random(seed).sample(ids, max_samples))
        self.ids: List[str] = ids
        if not self.ids:
            raise RuntimeError(f"No <id>/image.jpg samples under {self.root}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Dict:
        image_id = self.ids[idx]
        folder = self.root / image_id
        size = (self.image_size, self.image_size)

        img = cv2.imread(str(folder / "image.jpg"), cv2.IMREAD_COLOR)  # BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)  # square, full image

        # Detect + crop a solid frame, then resize the content back to image_size so
        # DCF runs on the framed-out object. crop_box records where the content sat.
        if self.remove_frame:
            t, b, l, r = detect_frame_crop(img)
        else:
            t, b, l, r = 0, self.image_size, 0, self.image_size
        content = cv2.resize(img[t:b, l:r], size, interpolation=cv2.INTER_AREA)
        image = torch.from_numpy(np.moveaxis(content, -1, 0)).to(torch.float32) / 255.0

        mask = cv2.imread(str(folder / "mask.jpg"), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        mask = torch.from_numpy((mask > 127).astype(np.uint8))  # full-image GT

        crop_box = torch.tensor([t, b, l, r], dtype=torch.long)
        return {"image": image, "mask": mask, "image_id": image_id, "crop_box": crop_box}


def _collate(batch: List[Dict]) -> Dict:
    return {
        "image": torch.stack([b["image"] for b in batch]),       # (B, C, H, W)
        "mask": torch.stack([b["mask"] for b in batch]),         # (B, H, W)
        "crop_box": torch.stack([b["crop_box"] for b in batch]),  # (B, 4) = t,b,l,r
        "image_id": [b["image_id"] for b in batch],
    }


def build_dataloader(
    root: Path,
    image_size: int = 512,
    batch_size: int = 4,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
    shuffle: bool = False,
    remove_frame: bool = True,
    seed: Optional[int] = 1234,
) -> DataLoader:
    dataset = ForegroundDataset(
        root, image_size=image_size, max_samples=max_samples,
        remove_frame=remove_frame, seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        collate_fn=_collate,
        pin_memory=torch.cuda.is_available(),
    )


def make_init_contours(
    batch_size: int,
    image_size: int,
    shape: str = "circle",
    size: float = 0.5,
    nb_nodes: int = 200,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Build the initial contour, normalized to [0, 1], shape ``(B, 1, nb_nodes, 2)``."""
    contour_init, _ = define_contour_init(n=image_size, shape=shape, size=size)
    cleaner = CleanContours()
    contour_init = cleaner.interpolate(contour_init, nb_nodes).clip(0, 1)
    contour = torch.tensor(contour_init, dtype=torch.float32)[None, None]  # (1, 1, K, 2)
    return contour.repeat(batch_size, 1, 1, 1).to(device)


def contours_to_masks(final_contours: np.ndarray, image_size: int) -> np.ndarray:
    """Rasterize DCF final contours ``(B, K, 2)`` (pixel coords) to binary masks ``(B, H, W)``.

    Matches the repo's own convention (cv2.fillPoly on integer (x, y) points).
    """
    masks = np.zeros((final_contours.shape[0], image_size, image_size), dtype=np.uint8)
    for i, contour in enumerate(final_contours):
        pts = np.asarray(contour).reshape(-1, 1, 2).astype(np.int32)
        cv2.fillPoly(masks[i], [pts], 1)
    return masks


def remap_masks_to_full(
    content_masks: np.ndarray, crop_boxes: np.ndarray, image_size: int
) -> np.ndarray:
    """Undo the frame crop: place each content-space mask back into the full canvas.

    ``content_masks`` is ``(B, image_size, image_size)`` (DCF output in cropped space);
    ``crop_boxes`` is ``(B, 4)`` of ``(top, bottom, left, right)`` in full image_size
    coords. Returns ``(B, image_size, image_size)`` masks aligned with the full image
    (zeros over the cropped-out frame band). A full-image box is an identity remap.
    """
    full = np.zeros((content_masks.shape[0], image_size, image_size), dtype=np.uint8)
    for i, (t, b, l, r) in enumerate(np.asarray(crop_boxes).astype(int)):
        if (t, b, l, r) == (0, image_size, 0, image_size):
            full[i] = content_masks[i]
            continue
        resized = cv2.resize(content_masks[i], (r - l, b - t), interpolation=cv2.INTER_NEAREST)
        full[i, t:b, l:r] = resized
    return full
