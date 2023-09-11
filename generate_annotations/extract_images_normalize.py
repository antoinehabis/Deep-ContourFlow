from utils import *
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from openslide import OpenSlide
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import tifffile


### Extraction of all the other patches and normalization
annotations = pd.read_csv(
    os.path.join(path_annotations, "annotations.csv"), index_col=0
)
n = annotations.shape[0]
annotations = annotations.replace(["dilated_tubule", "fake_tubule"], [1, 0])
filenames = np.unique(list(annotations["image"]))

coordinates_start = {}

for filename in filenames:
    thresh = find_thresh(filename, percentile=90)
    slide_path = os.path.join(path_slides, filename)
    im = OpenSlide(slide_path)
    anns = annotations[annotations["image"] == filename]

    for row in tqdm(anns.iterrows()):
        coordinates, term = row_to_coordinates(row)

        img, contour_true = process_coord_get_image(coordinates, im=im, margin=100)

        mask = cv2.fillPoly(
            np.zeros((img.shape[0], img.shape[1])), contour_true[None].astype(int), 1, 0
        ).astype(int)

        img, contour_init = retrieve_img_contour(img=img, thresh=thresh, mask=mask)
        filename = row_to_filename(row)

        tifffile.imsave(os.path.join(os.path.join(path_data, "masks"), filename), mask)
        tifffile.imsave(os.path.join(os.path.join(path_data, "images"), filename), img)

        coordinates_start[filename] = interpolate(contour_init, 100)
np.save(os.path.join(path_data, "contour_init.npy"), coordinates_start)
