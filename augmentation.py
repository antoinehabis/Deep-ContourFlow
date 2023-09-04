
from config import *

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
    img = img/255
    return img, mask 


