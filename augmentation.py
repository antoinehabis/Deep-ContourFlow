
from config import *
from numba import jit
from skimage.filters import gaussian
# import albumentations as A
# from tps import ThinPlateSpline
dim = 512

# @jit
# def mapping_coordinates(coordinates, img):
    
#     n,m = img.shape[:2]
#     new_img = np.zeros(img.shape)
    
#     for i in range(n):
#         for j in range(m): 
            
#             coord = coordinates[i,j]
#             x = coord[0]
#             y = coord[1]
#             if x >= 0 and x <m:
#                 if y >= 0 and y <n:
#                     new_img[i, j] = img[y, x]
#     return new_img
   



# def elastic_transform(img,
#                       mask,
#                       std):
    
#     tps = ThinPlateSpline()

    
#     l,c = img.shape[:2]

#     n = 4
    
#     mesh = np.stack(np.meshgrid(np.arange(c), np.arange(l)),-1)
#     X_c = np.concatenate((np.random.randint(0, c, (n, 1)), np.random.randint(0, l, (n, 1))), axis = -1)

#     mean = 0

#     X_t = X_c + np.round(np.random.randn(n*2)*2).reshape(n,2)
#     tps.fit(X_c, X_t)

#     # # Transform new points
#     Y = tps.transform(mesh.reshape(-1,2))
#     coordinates = np.round(Y.reshape(l,c, 2)).astype(int)
#     return mapping_coordinates(coordinates, img), mapping_coordinates(coordinates, mask)



def augmentation(img,
                 mask):
    
    
    img = (img*255).astype(np.uint8)
    mask_shape = mask.shape
    
    if len(mask_shape)==2:
        mask = np.expand_dims(mask, -1)
        
    ps = np.random.random(10)
    if ps[0]>1/4 and ps[0]<1/2:
        img, mask = np.rot90(img,axes = [0,1], k=1), np.rot90(mask,axes = [0,1], k=1)
        
    if ps[0]>1/2 and ps[0]<3/4:
        img, mask = np.rot90(img,axes = [0,1], k=2), np.rot90(mask,axes = [0,1], k=2)
        
    if ps[0]>3/4 and ps[0]<1:
        img, mask = np.rot90(img,axes = [0,1], k=3), np.rot90(mask,axes = [0,1], k=3)
        
    if ps[1]>0.5:
        img, mask = np.flipud(img), np.flipud(mask)
        
    if ps[2]>0.5:
        img, mask = np.fliplr(img), np.fliplr(mask)
    mask = mask[:,:]
#     img, mask = elastic_transform(img,mask,std=0)
    img = img/255
    return img, mask 


