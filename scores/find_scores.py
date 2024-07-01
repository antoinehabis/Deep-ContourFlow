import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from config import *
from algorithms.dcf_distance_map import DCF
from algorithms.utils_dilated_tubules import *
from utils import *
import pandas as pd
import tifffile
from tqdm import tqdm 

path_images = os.path.join(path_data,'images')
path_masks = os.path.join(path_data,'masks')
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

slides_already_processed = list(np.unique(df['slide']))
all_filenames = np.unique(list(annotations['image']))
filenames_to_process = list(set(all_filenames) - set(slides_already_processed))



for filename in tqdm(filenames_to_process):
    
    annotations = pd.read_csv(os.path.join(path_annotations,'annotations.csv'), index_col=0)
    annotations = annotations.replace(['dilated_tubule', 'fake_tubule'],[1,0])
    annotations = annotations[annotations['image'] == filename]
    annotations_anchor = annotations[annotations['term'] == 1]
    annotations_anchor = annotations_anchor.sample(frac = 1).head(10)

    for index0, row0 in annotations_anchor.iterrows():

        dac = DCF(nb_points = 100,
                  n_epochs = 200,
                  nb_augment = 100,
                  isolines = np.array([0., 1.]),
                  learning_rate = 5e-2,
                  clip = 1e-1,
                  sigma = 5,
                  weights = 0.9,
                  exponential_decay = 1.,
                  thresh = 1e-2)

        filename_img_anchor = row_to_filename(row0)
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
            img = tifffile.imread(os.path.join(path_images,filename_img))
            term = row1.term
            contour_init = contour_inits[filename_img]

            C0 = preprocess_contour(contour_init,img)
            shape_fin, score, tots, energies = dac.predict(img,C0)

            x = np.argmin(tots)

            energie_fin = tots[x]
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
    df.to_csv('scores.csv')    
