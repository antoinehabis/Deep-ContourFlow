"""
Utilitaires pour les algorithmes DCF.

Ce module contient des fonctions utilitaires pour le post-processing
et d'autres opérations communes aux algorithmes DCF.
"""

import logging
import multiprocessing as mp
from typing import Tuple

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt, label

logger = logging.getLogger(__name__)


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
        mask_grabcut[distance_map > 0.3] = cv2.GC_FGD
        mask_grabcut[(distance_map > 0.3) & (distance_map <= 0.8)] = cv2.GC_PR_FGD
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
        img_list = [img[i] for i in range(img.shape[0])]
        args_list = [(img_list[i], final_contours[i]) for i in range(len(img_list))]
        with mp.Pool(processes=min(mp.cpu_count(), len(args_list))) as pool:
            results = pool.map(process_grabcut_single_helper, args_list)

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
        results = []
        for i in range(img.shape[0]):
            try:
                result = process_grabcut_single_helper((img[i], final_contours[i]))
                results.append(result)
            except Exception as e:
                logger.warning(f"Error processing image {i} with GrabCut: {e}")
                results.append(final_contours[i])

        logger.info("GrabCut post-processing completed with sequential processing")
        return np.array(results)

    except Exception as e:
        logger.error(f"Error in sequential GrabCut post-processing: {e}")
        return final_contours  # Return original contours if processing fails
