import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import *
from scipy.interpolate import interp1d
import numpy as np
from isect_segments_bentley_ottmann import poly_point_isect

def interpolate(shape, n):
    distance = np.cumsum(np.sqrt(np.sum(np.diff(shape, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0) / distance[-1]
    alpha = np.linspace(0, 1, n)
    shape = interp1d(distance, shape, kind="linear", axis=0)(alpha)

    return shape


def augmentation(img, mask):
    img = (img * 255).astype(np.uint8)
    mask_shape = mask.shape

    if len(mask_shape) == 2:
        mask = np.expand_dims(mask, -1)

    ps = np.random.random(10)
    if ps[0] > 1 / 4 and ps[0] < 1 / 2:
        img, mask = np.rot90(img, axes=[0, 1], k=1), np.rot90(mask, axes=[0, 1], k=1)

    if ps[0] > 1 / 2 and ps[0] < 3 / 4:
        img, mask = np.rot90(img, axes=[0, 1], k=2), np.rot90(mask, axes=[0, 1], k=2)

    if ps[0] > 3 / 4 and ps[0] < 1:
        img, mask = np.rot90(img, axes=[0, 1], k=3), np.rot90(mask, axes=[0, 1], k=3)

    if ps[1] > 0.5:
        img, mask = np.flipud(img), np.flipud(mask)

    if ps[2] > 0.5:
        img, mask = np.fliplr(img), np.fliplr(mask)
    mask = mask[:, :]
    img = img / 255
    return img, mask


def delete_loops(contour):
    tuples = poly_point_isect.isect_polygon_include_segments(contour)

    if len(tuples)>0:
        indices = np.arange(contour.shape[0])
        for isect, segment in tuples:
            index1 = np.where(np.all(contour == segment[0][0], axis=-1))[0][0]
            index2 = np.where(np.all(contour == segment[1][0], axis=-1))[0][0]
            min_index = np.minimum(index1,index2)
            max_index = np.maximum(index1,index2)

            if np.abs(index1 - index2) / contour.shape[0] < 0.5:
                new_indices = np.concatenate(
                    [np.arange(min_index), np.arange(max_index, contour.shape[0])]
                )
            else:
                new_indices = np.concatenate([np.arange(min_index+1, max_index-1)])
            indices = np.intersect1d(indices, new_indices)
        contour = contour[indices]
    return contour