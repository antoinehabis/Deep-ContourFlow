
import os
from pathlib import Path

path_data = '/home/ahabis/sshfs_zeus/DILATED_TUBULES/data'
path_dac = str(Path(__file__).resolve().parent)
path_slides = os.path.join(path_data,'slides')
path_annotations = os.path.join(path_data,'annotations')
path_images = os.path.join(path_data,'images')
path_masks = os.path.join(path_data,'masks')
path_scores = os.path.join(path_dac,'scores')