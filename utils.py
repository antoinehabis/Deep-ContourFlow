import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import *
from skimage.measure import label
from scipy.interpolate import interp1d
from histolab.slide import Slide
import numpy as np
import cv2
from scipy.ndimage import binary_closing, binary_opening
from skimage.morphology import disk


def row_to_filename(row):
    filename = row.image.split(".")[0] + "_" + str(row.id) + ".tif"
    return filename


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


def interpolate(shape, n):
    distance = np.cumsum(np.sqrt(np.sum(np.diff(shape, axis=0) ** 2, axis=1)))
    distance = np.insert(distance, 0, 0) / distance[-1]
    alpha = np.linspace(0, 1, n)
    shape = interp1d(distance, shape, kind="linear", axis=0)(alpha)

    return shape



def augmentation(img,
                 mask):
    
    
    img = (img*255).astype(np.uint8)
    mask_shape = mask.shape
    
    if len(mask_shape)==2:
        mask = np.expand_dims(mask, -1)
        
    ps = np.random.random(10)
    if ps[0]>1/4 and ps[0]<1/2:
        img, mask = np.rot90(img,axes = [0,1], k=1), np.rot90(mask,axes = [0,1], k=1)
        
    if ps[0]>1/2 and ps[0]<3/4:
        img, mask = np.rot90(img,axes = [0,1], k=2), np.rot90(mask,axes = [0,1], k=2)
        
    if ps[0]>3/4 and ps[0]<1:
        img, mask = np.rot90(img,axes = [0,1], k=3), np.rot90(mask,axes = [0,1], k=3)
        
    if ps[1]>0.5:
        img, mask = np.flipud(img), np.flipud(mask)
        
    if ps[2]>0.5:
        img, mask = np.fliplr(img), np.fliplr(mask)
    mask = mask[:,:]
    img = img/255
    return img, mask 


def delete_loops(contour,
                 shape):
    
    contour = (contour*shape).astype(int)
    zeros = np.zeros(shape)
    new_img = cv2.fillPoly(zeros,[contour],1)
    new_img = binary_opening(new_img,disk(2))

    label_=  label(new_img,connectivity=1)
    uniques, counts = np.unique(label_,
                                return_counts = True)

    biggest = uniques[np.argsort(counts)[-2]]

    contour = np.squeeze(cv2.findContours((label_ == biggest).astype(int), 
                method = cv2.RETR_TREE,
                mode=cv2.CHAIN_APPROX_SIMPLE,
                )[0][0])/shape
    
    return contour