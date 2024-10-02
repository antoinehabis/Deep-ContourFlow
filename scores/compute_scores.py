import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import *
from algorithms.dcf_distance_map import DCF
from utils import  row_to_filename, preprocess_contour
from tqdm import tqdm
import pandas as pd
from skimage.morphology import disk
import tifffile
import numpy as np
import cv2

annotations = pd.read_csv(
    os.path.join(path_annotations, "annotations.csv"), index_col=0
)
contour_inits = np.load(
    os.path.join(path_data, "contour_init.npy"), allow_pickle=True
).item()

df = pd.DataFrame(
    columns=[
        "slide",
        "nb_support",
        "DICE(%)",
        "IOU(%)",
        "gt",
        "score",
        "nb_query",
    ]
)
try:
    df = pd.read_csv(os.path.join(path_scores, "scores.csv"), index_col=0)
except:
    pass

slides_already_processed = list(np.unique(df["slide"]))

all_filenames = np.unique(list(annotations["slide"]))
filenames_to_process = list(set(all_filenames) - set(slides_already_processed))


def compute_scores_filename(filename, df):
    annotations = pd.read_csv(
        os.path.join(path_annotations, "annotations.csv"), index_col=0
    )
    annotations = annotations.replace(["dilated_tubule", "fake_tubule"], [1, 0])

    ### We extract only the annotations of a given slide

    annotations = annotations[annotations["slide"] == filename]

    ### We take only the dilated tubule of the slide

    annotations_support = annotations[annotations["term"] == 1]
    annotations_support = annotations_support.sample(frac=1).head(10)

    for _, row0 in annotations_support.iterrows():
        dcf = DCF(
            nb_points=100,
            n_epochs=300,
            nb_augment=100,
            isolines=np.array([0.0, 1.0]),
            learning_rate=5e-2,
            clip=1e-1,
            sigma=5,
            weights=0.9,
            exponential_decay=0.999,
            thresh=1e-2,
        )

        filename_img_support = row_to_filename(row0)

        print("support_filename", filename_img_support)

        img_support = tifffile.imread(os.path.join(path_images, filename_img_support))
        mask_support = tifffile.imread(os.path.join(path_masks, filename_img_support))
        contour_support = np.squeeze(
            cv2.findContours(
                mask_support.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )[0][0]
        )

        dcf.fit(img_support, contour_support, augment=True)

        for _, row1 in annotations.iterrows():
            filename_img = row_to_filename(row1)
            img = tifffile.imread(os.path.join(path_images, filename_img))
            term = row1.term
            contour_init = contour_inits[filename_img]

            C0 = preprocess_contour(contour_init, img)
            shape_fin, score, tots, energies = dcf.predict(img, C0)

            x = np.argmin(tots)

            contour_pred = shape_fin[x]

            img_true = tifffile.imread(os.path.join(path_masks, filename_img))
            img_pred = cv2.fillPoly(np.zeros(img.shape[:-1]), contour_pred[None], 1)
            num_DICE = 2 * np.sum(img_true * img_pred)
            denom_DICE = np.sum(img_true) + np.sum(img_pred)
            num_IOU = np.sum(img_true * img_pred)
            denom_IOU = np.sum(np.maximum(img_true, img_pred))

            DICE = (num_DICE / denom_DICE) * 100
            IOU = (num_IOU / denom_IOU) * 100

            df = df.append(
                {
                    "slide": row0.slide,
                    "nb_support": row0.id,
                    "DICE(%)": np.round(DICE, 2),
                    "IOU(%)": np.round(IOU, 2),
                    "gt": term,
                    "score": score,
                    "nb_query": row1.id,
                },
                ignore_index=True,
            )

            dice = np.array(list(df["DICE(%)"]))
            gt = np.array(list(df["gt"]))
            print(str(np.sum(dice * gt) / np.sum(gt)))
    df.to_csv(os.path.join(path_scores, "scores.csv"))

    return df

for filename in tqdm(filenames_to_process):
    df = compute_scores_filename(filename, df)
