# Deep-Active-Contour

In This repository you can find the code for both: unsupervised Deep-ContourFlow and One shot learning Deep-ContourFlow.

![Alt text](./folder_images_paper/real_life_images.png "Unsupervised DCF: evolution of the contour on four real-life images when varying the initial contour")
![Alt text](./folder_images_paper/skin_lesions.png "Unsupervised DCF: evolution of the contour on three skin lesions from Skin Cancer MNIST: HAM10000")
![Alt text](./folder_images_paper/tumor_region.png "Unsupervised DCF: evolution of the contour on two histology images.")

To use this repository please first install torch-contour:
```
$pip install torch_contour
```

## Unsupervised Deep Active Contour:

To use Unsupervised DCF just add your image in ```images_test_unsupervised_dcf``` and run the algorithm using the notebook: ```unsupervised_dcf.ipynb```


## One shot learning: Application on Dilated Tubules

To replicate the results in the paper ... follow the steps:

### 0. Download AIDPATH kidney dataset:

1. Register to AIDPATHDB using the link https://mitel.dimi.uniud.it/aidpath-db/app/login.php
2. Donwload the kidney dataset of AIDPATHDB and put the images in ```slides```
3. The manual annotations are available in the github in the ```generate_annotations``` folder
   
### 1. Extract images:

 ```
cd ./generate_annotations
python extract_images.py
```

This code uses ```annotations.csv``` and the slides extracted in #0 in folder ```slides``` to extract all the patches centered on dilated tubule and "false dilated tubule" in the slides.
The code will create:

1. A folder 'masks' with the ground truth masks of the 2 class: "False" and  "True" dilated tubule.
2. A folder 'images' with the correponding images.
3. A file contour_init.npy the initial contour corresponding to the detection of white for each image.

### 2. Run and get results:

```
cd ./scores
python compute_scores.py
```

This code take 10 random dilated tubule in each slide and fit the One shot learning DCF and predict for all other object in the slide.
It creates a csv file containing all the scores.



   



