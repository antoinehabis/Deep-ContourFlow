import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import *
from skimage.measure import label
from scipy.interpolate import interp1d
import numpy as np
import cv2
from scipy.ndimage import binary_closing
from skimage.morphology import disk
from histolab.slide import Slide



def row_to_filename(row):
    filename = row.slide.split(".")[0] + "_" + str(row.id) + ".tif"
    return filename


def define_contour_init(img, center, axes, angle = 0):
  # major, minor axes
    start_angle = 0
    end_angle = 360
    color = 1
    thickness = -1

    # Draw a filled ellipse on the input image
    mask = cv2.ellipse(
        np.zeros(img.shape[:-1]),
        center,
        axes,
        angle,
        start_angle,
        end_angle,
        color,
        thickness,
    ).astype(np.uint8)
    contour = np.squeeze(
        cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0][0]
    )
    return contour, mask

def find_thresh(filename, percentile):
    img = Slide(os.path.join(path_slides, filename), processed_path="")
    arr = img.resampled_array(scale_factor=4)
    gray = np.mean(arr, -1)
    ret2, th2 = cv2.threshold(
        np.mean(arr, -1).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    new_img = binary_closing(1 - th2 / 255, disk(9)).astype(bool)
    return np.percentile(gray[new_img], percentile) / 255

def row_to_coordinates(row):
    class_ = row.term
    string = row.location
    string = string.replace("POLYGON ", "")
    string = string.replace(", ", "),(")
    string = string.replace(" ", ",")
    coordinates = np.array(eval(string))
    return coordinates, class_

def preprocess_contour(contour_init,
                       img):
    img = cv2.fillPoly(np.zeros(img.shape[:-1]), [contour_init.astype(int)], 1)
    img = binary_closing(img,disk(5))      
    contour_init = np.squeeze(cv2.findContours(img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0][0])
    return contour_init



def process_coord_get_image(coord, im, margin=100):
    coord_tmp = coord.copy()

    coord_tmp[:, 1] = im.dimensions[1] - coord_tmp[:, 1]
    coord_min = coord_tmp - np.min(coord_tmp, 0)
    x_min, y_min = np.min(coord_tmp, 0).astype(int) - margin
    x_max, y_max = np.max(coord_tmp, 0).astype(int) + margin

    img = np.array(
        im.read_region(
            location=[x_min, y_min], level=0, size=[x_max - x_min, y_max - y_min]
        )
    )[:, :, :-1]
    contour = (coord_min + margin).astype(int)
    return img, contour


def retrieve_img_contour(img, thresh, mask):
    img = img / np.max(img)
    mean = np.mean(img, -1)
    l, c = np.array(img.shape[:-1]) // 2

    x = label(binary_closing(mean > thresh, disk(5)))
    lab = x * mask
    uniques, counts = np.unique(lab[lab > 0], return_counts=True)
    arg = np.argsort(counts)[-1]

    white = (x == uniques[arg]).astype(int)
    shapes = cv2.findContours(
        white.astype(np.uint8),
        method=cv2.RETR_TREE,
        mode=cv2.CHAIN_APPROX_SIMPLE,
    )[0]

    return (img * 255).astype(np.uint8), np.squeeze(shapes[0])
