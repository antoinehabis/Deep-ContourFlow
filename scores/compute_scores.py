import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import *
from dac_distance_map import DAC
# from utils import *
from scipy.ndimage import binary_dilation
from tqdm import tqdm
import pandas as pd
from scipy.ndimage import binary_closing
from skimage.morphology import disk
import tifffile
import numpy as np
import cv2

def row_to_coordinates(row):
    class_ = row.term
    string = row.location
    string = string.replace("POLYGON ", "")
    string = string.replace(", ", "),(")
    string = string.replace(" ", ",")
    coordinates = np.array(eval(string))
    return coordinates, class_

def row_to_filename(row):
    filename = row.image.split(".")[0] + "_" + str(row.id) + ".tif"
    return filename

annotations = pd.read_csv(os.path.join(path_annotations,'annotations.csv'), index_col=0)
contour_inits = np.load(os.path.join(path_data, 'contour_init.npy'), allow_pickle=True).item()

df = pd.DataFrame(columns = ['slide',
                             'nb_anchor',
                             'DICE(%)',
                             'IOU(%)',
                             'gt',
                             'score',
                             'num_DICE',
                             'num_IOU',
                             'denom_DICE',
                             'denom_IOU',
                             'nb_image'])
try:
        df = pd. read_csv(os.path.join(path_scores,'scores.csv'), index_col=0)
except:
        pass

slides_already_processed = list(np.unique(df['slide']))
all_filenames = np.unique(list(annotations['image']))
filenames_to_process = list(set(all_filenames) - set(slides_already_processed))

def preprocess_contour(contour_init,
                       img,
                       AREA_LIMIT = 10000):
                       
    img = cv2.fillPoly(np.zeros(img.shape[:-1]), [contour_init.astype(int)], 1)
    img = binary_closing(img,disk(5))      
    contour_init = np.squeeze(cv2.findContours(img.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0][0])
    return contour_init

for filename in tqdm(filenames_to_process):
    
    annotations = pd.read_csv(os.path.join(path_annotations,'annotations.csv'), index_col=0)
    annotations = annotations.replace(['dilated_tubule', 'fake_tubule'],[1,0])
    annotations = annotations[annotations['image'] == filename]
    annotations_anchor = annotations[annotations['term'] == 1]
    annotations_anchor = annotations_anchor.sample(frac = 1).head(10)

    for index0, row0 in annotations_anchor.iterrows():

        dac = DAC(nb_points = 100,
                  n_epochs = 300,
                  nb_augment = 100,
                  isolines = np.array([0., 1.]),
                  learning_rate = 5e-2,
                  clip = 1e-1,
                  sigma = 5.,
                  weights = 0.9,
                  exponential_decay = 0.999,
                  thresh = 1e-2)

        filename_img_anchor = row_to_filename(row0)
        
        print('anchor_filename',filename_img_anchor)

        img_anchor = tifffile.imread(os.path.join(path_images,filename_img_anchor))
        mask_anchor = tifffile.imread(os.path.join(path_masks,filename_img_anchor))
        contour_anchor = np.squeeze(cv2.findContours(mask_anchor.astype(np.uint8),
                                 cv2.RETR_TREE,
                                 cv2.CHAIN_APPROX_SIMPLE)[0][0])
        
        contour_anchor = preprocess_contour(contour_anchor,img_anchor)
        dac.fit(img_anchor,
                contour_anchor,
                augment = True)
        
        for index1, row1 in annotations.iterrows():
        
            filename_img = row_to_filename(row1)
            if filename_img != '20163_2692503.tif':

                img = tifffile.imread(os.path.join(path_images,filename_img))
                term = row1.term
                contour_init = contour_inits[filename_img]

                C0 = preprocess_contour(contour_init,img)
                shape_fin, score, tots, energies = dac.predict(img,C0)

                x = np.argmin(tots)

                energie_fin = energies[x]
                contour_pred = shape_fin[x]

                img_true = tifffile.imread(os.path.join(path_masks,filename_img))
                img_pred = cv2.fillPoly(np.zeros(img.shape[:-1]), contour_pred[None], 1)
                num_DICE = 2 * np.sum(img_true * img_pred)
                denom_DICE = (np.sum(img_true) + np.sum(img_pred))

                num_IOU = np.sum(img_true * img_pred)
                denom_IOU = np.sum(np.maximum(img_true, img_pred))

                DICE = (num_DICE / denom_DICE)*100
                IOU = (num_IOU / denom_IOU)*100

                df = df.append({'slide':row0.image,
                                'nb_anchor':row1.id,
                                'DICE(%)':np.round(DICE,2),
                                'IOU(%)':np.round(IOU,2),
                                'gt':term,
                                'score':score,
                                'num_DICE':num_DICE,
                                'num_IOU':num_IOU,
                                'denom_DICE':denom_DICE,
                                'denom_IOU':denom_IOU,
                                'energy':energie_fin,
                                'nb_image':row0.id,
                                }, ignore_index = True)
                                
                dice = np.array(list(df['DICE(%)']))
                gt = np.array(list(df['gt']))
                print(str(np.sum(dice*gt)/np.sum(gt)))

    df.to_csv(os.path.join(path_scores,'scores.csv'))
