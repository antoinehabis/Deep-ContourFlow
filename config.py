
import os
from pathlib import Path

path_data = os.getenv('PATH_DATA_DILATE_TUBULES')
path_dcf = str(Path(__file__).resolve().parent)
path_slides = os.path.join(path_data,'slides')
path_annotations = os.path.join(path_dcf,'generate_annotations')
path_images = os.path.join(path_data,'images')
path_masks = os.path.join(path_data,'masks')
path_scores = os.path.join(path_dcf,'scores')