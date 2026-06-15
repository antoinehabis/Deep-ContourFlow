"""Utility functions for DCF algorithms (post-processing and shared operations)."""

import logging
import multiprocessing as mp
from typing import Tuple
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, label

logger = logging.getLogger(__name__)


def _resample_closed_contour(c: np.ndarray, k: int) -> np.ndarray:
    """Resample a polygon ``(N, 2)`` to exactly ``k`` points, evenly spaced along
    its perimeter. Lets GrabCut outputs (which have a variable point count) be
    stacked into a homogeneous ``(B, k, 2)`` array."""
    c = np.asarray(c, dtype=np.float64).reshape(-1, 2)
    if len(c) < 2:
        return np.zeros((k, 2), dtype=np.float64)
    cc = np.vstack([c, c[:1]])  # close the loop
    seg = np.sqrt((np.diff(cc, axis=0) ** 2).sum(1))
    d = np.concatenate([[0.0], np.cumsum(seg)])
    total = d[-1]
    if total <= 0:
        return np.repeat(c[:1], k, axis=0)
    t = np.linspace(0.0, total, k, endpoint=False)
    x = np.interp(t, d, cc[:, 0])
    y = np.interp(t, d, cc[:, 1])
    return np.stack([x, y], axis=1)


def _contour_area(c: np.ndarray) -> float:
    """Shoelace area (px²) of a polygon ``(N, 2)``; used to skip degenerate
    contours before GrabCut (an empty/near-zero mask can crash cv2.grabCut)."""
    c = np.asarray(c, dtype=np.float64).reshape(-1, 2)
    if len(c) < 3:
        return 0.0
    x, y = c[:, 0], c[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def process_grabcut_single_helper(args: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """
    Helper function for multiprocessing GrabCut processing.

    Args:
        args: Tuple containing (img_np, contour)

    Returns:
        Processed contour
    """
    img_np, contour = args
    try:
        img_np = np.moveaxis(img_np, 0, -1)  # (C, H, W) -> (H, W, C)
        img_np = (img_np * 255).astype(np.uint8)
        mask = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)
        if len(contour.shape) == 2:
            contour_for_fill = contour.reshape(-1, 1, 2).astype(int)
        else:
            contour_for_fill = contour.astype(int)

        cv2.fillPoly(mask, [contour_for_fill], 1)

        distance_map = distance_transform_edt(mask)
        distance_map = distance_map / np.max(distance_map)
        distance_map_outside = distance_transform_edt(1 - mask)
        distance_map_outside = distance_map_outside / np.max(distance_map_outside)

        mask_grabcut = np.full(mask.shape, cv2.GC_PR_BGD, dtype=np.uint8)
        mask_grabcut[distance_map > 0.2] = cv2.GC_FGD
        mask_grabcut[(distance_map > 0.2) & (distance_map <= 0.8)] = cv2.GC_PR_FGD
        mask_grabcut[distance_map_outside > 0.8] = cv2.GC_BGD

        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        cv2.grabCut(
            img_np,
            mask_grabcut,
            None,
            bgdModel,
            fgdModel,
            5,
            cv2.GC_INIT_WITH_MASK,
        )

        result = np.where(
            (mask_grabcut == cv2.GC_FGD) | (mask_grabcut == cv2.GC_PR_FGD), 1, 0
        ).astype(np.uint8)

        labeled_array, num_features = label(result)
        if num_features > 0:
            largest_cc = np.argmax(np.bincount(labeled_array.flat)[1:]) + 1
            result = (labeled_array == largest_cc).astype(np.uint8)

        contours, _ = cv2.findContours(
            result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            return largest_contour.reshape(-1, 2)
        else:
            return contour

    except Exception as e:
        logger.error(f"Error in single GrabCut processing: {e}")
        return contour


def apply_grabcut_postprocessing_parallel(
    img: np.ndarray, final_contours: np.ndarray
) -> np.ndarray:
    """
    Apply GrabCut post-processing with parallel processing for better performance.

    Args:
        img: Input image tensor (B, C, H, W)
        final_contours: Final contours from DCF (B, K, 2)

    Returns:
        Refined contours after GrabCut processing
    """
    try:
        k = np.asarray(final_contours).shape[1]
        n = img.shape[0]
        # Default: keep the (resampled) original contour. Skip GrabCut for
        # degenerate/near-zero-area contours, which can crash cv2.grabCut.
        results = [
            _resample_closed_contour(np.asarray(final_contours[i]), k) for i in range(n)
        ]
        process_idx = [i for i in range(n) if _contour_area(final_contours[i]) >= 16.0]
        if process_idx:
            args_list = [(img[i], final_contours[i]) for i in process_idx]
            with mp.Pool(processes=min(mp.cpu_count(), len(args_list))) as pool:
                refined = pool.map(process_grabcut_single_helper, args_list)
            for i, r in zip(process_idx, refined):
                # Resample each GrabCut contour to k points so the batch stays a
                # homogeneous (B, k, 2) array (the variable point count used to
                # raise "inhomogeneous shape" and silently skip GrabCut entirely).
                results[i] = _resample_closed_contour(r, k)

        logger.info("GrabCut post-processing completed with parallel processing")
        return np.array(results)

    except Exception as e:
        logger.error(f"Error in parallel GrabCut post-processing: {e}")
        return final_contours  # Return original contours if parallel processing fails


def apply_grabcut_postprocessing_sequential(
    img: np.ndarray, final_contours: np.ndarray
) -> np.ndarray:
    """
    Apply GrabCut post-processing with sequential processing for better stability.

    Args:
        img: Input image tensor (B, C, H, W)
        final_contours: Final contours from DCF (B, K, 2)

    Returns:
        Refined contours after GrabCut processing
    """
    try:
        k = np.asarray(final_contours).shape[1]
        results = []
        for i in range(img.shape[0]):
            orig = _resample_closed_contour(np.asarray(final_contours[i]), k)
            # Skip GrabCut for degenerate contours (can crash cv2.grabCut).
            if _contour_area(final_contours[i]) < 16.0:
                results.append(orig)
                continue
            try:
                result = process_grabcut_single_helper((img[i], final_contours[i]))
                results.append(_resample_closed_contour(result, k))
            except Exception as e:
                logger.warning(f"Error processing image {i} with GrabCut: {e}")
                results.append(orig)

        logger.info("GrabCut post-processing completed with sequential processing")
        return np.array(results)

    except Exception as e:
        logger.error(f"Error in sequential GrabCut post-processing: {e}")
        return final_contours  # Return original contours if processing fails
