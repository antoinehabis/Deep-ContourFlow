from config import *
from utils import *


### Extraction of all the other patches and normalization
annotations = pd.read_csv(os.path.join(path_annotations,'annotations.csv'), index_col = 0)
n = annotations.shape[0]
annotations = annotations.replace(['dilated_tubule', 'fake_tubule'], [1,0])
filenames = np.unique(list(annotations['image']))

coordinates_start = {}

for filename in filenames : 
    thresh  = find_thresh(filename,percentile = 90)
    slide_path = os.path.join(path_slides, filename)
    im = OpenSlide(slide_path)
    anns = annotations[annotations['image']==filename]

    for index in tqdm(list(anns.index)):
    

        coordinates, term = index_to_coordinates(index,
                                                 annotations = anns)

        img, contour_true = process_coord_get_image(coordinates,
                                                    im = im,
                                                    margin = 100)

        mask = cv2.fillPoly(np.zeros((img.shape[0], img.shape[1])),contour_true[None].astype(int),1, 0).astype(int)

        img, contour_init = retrieve_img_contour(img=img,
                                                 thresh=thresh,
                                                 mask = mask)
        filename = index_to_filename(index, anns)
        
        tifffile.imsave(os.path.join(os.path.join(path_data,'masks'),filename),mask)
        tifffile.imsave(os.path.join(os.path.join(path_data,'images'),filename),img)

        coordinates_start[filename] = interpolate(contour_init,100)
np.save(os.path.join(path_data,'contour_init.npy'), coordinates_start)

