from config import *
from skimage.measure import label

def delete_loops(contour,
                 shape):
    
    contour = (contour*shape).astype(int)
    zeros = np.zeros(shape)
    new_img = cv2.fillPoly(zeros,[contour],1)
    new_img = binary_opening(new_img,disk(2))

    label_=  label(new_img,connectivity=1)
    uniques, counts = np.unique(label_,
                                return_counts = True)

    biggest = uniques[np.argsort(counts)[-2]]

    contour = np.squeeze(cv2.findContours((label_ == biggest).astype(int), 
                method = cv2.RETR_TREE,
                mode=cv2.CHAIN_APPROX_SIMPLE,
                )[0][0])/shape
    
    return contour