# Deep-Active-Contour

In This repository you can find the code for both: unsupervised Deep Active Contour and One shot learning distance map Deep active contour.

![alt text](https://github.com/antoinehabis/Deep-Active-Contour/blob/master/results_unsupervised_dac/flowers_evo.jpg)
![alt text](https://github.com/antoinehabis/Deep-Active-Contour/blob/master/results_unsupervised_dac/pineaples_evo.jpg)


## Unsupervised Deep Active Contour:

To use UDAC just add your image in ```images_test_unsupervised_dac``` and run the algorithm using the notebook: ```unsupervised_dac.ipynb```


## One shot learning: Application on dilated Tubules

To replicate the results in the paper ... follow the steps:

### 0. Download AIDPATH kidney dataset:

1. Register to AIDPATHDB using the link https://mitel.dimi.uniud.it/aidpath-db/app/login.php
2. Donwload the kidney dataset of AIDPATHDB and put the images in ```slides```
3. The manual annotations are available in the github in the ```annotations``` folder
   
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

This code take 10 random dilated tubule in each slide and fit the One shot learning DAC and predict for all other object in the slide.
It creates a csv file containing all the scores.



   



