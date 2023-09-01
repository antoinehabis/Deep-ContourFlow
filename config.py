
import os
import numpy as np 
import sys
import pandas as pd
import matplotlib.pyplot as plt
import torch
import tifffile
import cv2
from tqdm import tqdm
from PIL import Image
import torch
from openslide import OpenSlide
from glob import glob
from scipy.ndimage import binary_opening, binary_closing, binary_dilation
from skimage.morphology import disk
from numba import jit
import pickle



path_dac = '/home/ahabis/3-Deep_active_contour'

path_slides = os.path.join(path_dac,'slides')
path_annotations = os.path.join(path_dac,'annotations')
path_data = os.path.join(path_dac,'data')
path_images = os.path.join(path_data,'images')
path_masks = os.path.join(path_data,'masks')

dim = 512








