# Deep ContourFLow

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
[![Mail](https://img.shields.io/badge/Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:antoine.habis.tlcm@gmail.com)
[![Downloads](https://static.pepy.tech/badge/torch_contour/month)](https://pepy.tech/project/torch_contour)
[![Downloads](https://static.pepy.tech/badge/torch_contour)](https://pepy.tech/project/torch_contour)
[![ArXiv Paper](https://img.shields.io/badge/DOI-10.1038%2Fs41586--020--2649--2-blue)](
https://doi.org/10.48550/arXiv.2407.10696)

In This repository you can find the code for both: unsupervised Deep-ContourFlow and One shot learning Deep-ContourFlow.

if you use the the code please cite the following paper:
```
@misc{habis2024deepcontourflowadvancingactive,
      title={Deep ContourFlow: Advancing Active Contours with Deep Learning}, 
      author={Antoine Habis and Vannary Meas-Yedid and Elsa Angelini and Jean-Christophe Olivo-Marin},
      year={2024},
      eprint={2407.10696},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.10696}, 
}
```

![Alt text](./folder_images_paper/real_life_images.png "Unsupervised DCF: evolution of the contour on four real-life images when varying the initial contour")
![Alt text](./folder_images_paper/skin_lesions.png "Unsupervised DCF: evolution of the contour on three skin lesions from Skin Cancer MNIST: HAM10000")
![Alt text](./folder_images_paper/tumor_region.png "Unsupervised DCF: evolution of the contour on two histology images.")

To use this repository please first install torch-contour:
```
$pip install torch_contour>=1.3.0
```

## Unsupervised Deep ContourFLow:

To use Unsupervised DCF just add your image in ```images_test_unsupervised_dcf``` and run the algorithm using the notebook: ```unsupervised_dcf.ipynb```


## One shot learning: Application on Dilated Tubules

To replicate the results in the paper ... follow the steps:

### 0. Download AIDPATH kidney dataset:

1. Register to AIDPATHDB using the link https://mitel.dimi.uniud.it/aidpath-db/app/login.php
2. Create a folder where data will be saved and write its path in ```path_data``` inside the config file.
Donwload the kidney dataset of AIDPATHDB and put the images in a subfolder called ```slides```.


Manual annotations are available in the github in the ```generate_annotations``` folder.
   
### 1. Extract images:

 ```
cd ./generate_annotations
python extract_images.py
```

This code uses ```annotations.csv``` and the slides downloaded in #0 in folder ```slides``` to extract all the patches centered on each dilated tubule and "false dilated tubule".

The code will create in your data folder:

1. A subfolder 'masks' with the ground truth masks of the 2 class: "False" and  "True" dilated tubule.
2. A subfolder 'images' with the correponding images.
3. A file contour_init.npy that contains the initial contour corresponding to the detection of lumen for each patch.

### 2. Run and get results:

```
cd ./scores
python compute_scores.py
```

This code take 10 random dilated tubule in each slide and fit the One shot learning DCF and predict for all other object in the slide.
It creates a csv file containing all the scores.



   



