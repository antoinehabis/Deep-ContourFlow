import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from .utils_dilated_tubules import *
from torchvision.models import vgg16
from torch.nn import CosineSimilarity
from torchvision import transforms
from torchstain import MacenkoNormalizer
import torch
import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
from torch_contour.torch_contour import  Smoothing, CleanContours
from tqdm import tqdm
from utils import *

preprocess = transforms.Compose(
    [
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

feature_extractor = vgg16(pretrained=True).features
cos = CosineSimilarity(dim=0, eps=1e-08).cuda()


class DCF:
    def __init__(
        self,
        n_epochs=100,
        nb_augment=1,
        model=feature_extractor,
        isolines=np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0]),
        sigma=7,
        learning_rate=5e-2,
        clip=1e-1,
        exponential_decay=0.998,
        thresh=1e-2,
        weights=0.9,
    ):
        super(DCF, self).__init__()

        self.n_epochs = n_epochs
        self.nb_augment = nb_augment
        self.model = model
        self.activations = {}
        self.isolines = isolines
        self.model[3].register_forward_hook(self.get_activations(0))
        self.model[8].register_forward_hook(self.get_activations(1))
        self.model[15].register_forward_hook(self.get_activations(2))
        self.model[22].register_forward_hook(self.get_activations(3))
        self.model[29].register_forward_hook(self.get_activations(4))
        self.learning_rate = learning_rate
        self.clip = clip
        self.shapes = {}
        self.ed = exponential_decay
        self.isolines = torch.from_numpy(isolines).cuda()
        self.thresh = thresh
        self.smooth = Smoothing(sigma)
        self.ctf = Contour_to_isoline_features(256, self.isolines)
        self.dtf = Distance_map_to_isoline_features(self.isolines, halfway_value=0.5)
        self.normalizer = MacenkoNormalizer(backend="numpy")
        self.normalizer.HERef = np.array(
            [
                [0.47262014, 0.17700575],
                [0.79697804, 0.84033483],
                [0.37610664, 0.51235373],
            ]
        )
        self.normalizer.maxCRef = np.array([1.43072807, 0.98501085])
        self.weights = torch.tensor([weights, (1 - weights)], device="cuda")
        self.cleaner = CleanContours()

    def get_activations(self, name):
        def hook(model, input, output):

            device = input[0].device
            self.activations[name] = output.to(device)
        return hook


    def fit(self, img_support, polygon_support, augment=True):

        if img_support.dtype != 'torch.float32':
            raise Exception("tensor must be of type float32")
        
        img_support_array = np.transpose(img_support.cpu().detach().numpy()[0],[1,2,0])
        img_support_array, _ = self.normalizer.normalize((img_support_array*255).astype(np.uint8), stains=True)
        img_suppor_array = img_support_array/255
        mask_support_array = cv2.fillPoly(np.zeros(img_support_array.shape[:-1]), [polygon_support], 1)[None, None]
        
        with torch.no_grad():
            if augment == False:
                self.nb_augment = 1

            else:
                for i in tqdm(range(self.nb_augment)):

                    img_augmented, mask_augmented = augmentation(img_support_array, mask_support_array)
                    

                    distance_map_support = distance_transform_edt(mask_augmented[0,0])[None, None]
                    distance_map_support = distance_map_support / np.max(distance_map_support)
                    
                    tensor_mask_augmented = torch.from_numpy(mask_augmented).permute(2, 0, 1).unsqueeze(0)
                    tensor_augmented = torch.from_numpy(img_augmented).permute(2, 0, 1).unsqueeze(0)
                    tensor_distance_map_augmented = torch.from_numpy(tensor_distance_map_augmented).permute(2, 0, 1).unsqueeze(0)

                    _ = self.model(preprocess(tensor_augmented))
        
                    if i == 0:
                        self.features_isolines_support, self.features_mask_support = self.dtf(
                            self.activations, tensor_distance_map_augmented, tensor_mask_augmented, 
                            compute_features_mask=True
                        )

                    else:
                        tmp, tmp_mask = self.dtf(
                            self.activations, tensor_distance_map_augmented, tensor_mask_augmented, 
                            compute_features_mask=True
                        )

                        for j, u in enumerate(zip(tmp, tmp_mask)):
                            self.features_isolines_support[j] += u[0]
                            self.features_mask_support[j] += u[1]

                self.features_isolines_support = [u / self.nb_augment for u in self.features_isolines_support]
                self.features_anchor_mask = [
                    u / self.nb_augment for u in self.features_mask_support
                ]

    def compute_multiscale_multiisoline_loss(self,features_isolines, weights):

        tot = torch.zeros(len(self.activations), device="cuda")
        energies = torch.zeros((len(self.activations), len(self.isolines)))
        weights = torch.tensor(
            [1/(2)**i for i in range(len(self.activations))], device="cuda", dtype=torch.float32
        )
        weights = weights / torch.sum(weights)

        #### compute loss

        for j, feature in enumerate(features_isolines):
            difference_features = feature - self.features_isolines_support[j]
            tmp = torch.sqrt(torch.norm(difference_features, dim=1))
            energies[j] = tmp
            tot[j] = weights[j] * torch.mean(self.weights * tmp)

        loss = tot.flatten().mean(-1)

        return loss, energies
    def compute_similarity_score(self, features_mask_query):
        b = features_mask_query.shape[0]
        score = np.zeros((b,len(self.activations)))
        for i in range(len(self.activations)):
            cos = self.weights[i] * self.features_mask_support[i] @  features_mask_query[i].T/ (torch.linalg.norm(self.features_mask_support[i])* torch.linalg.norm(features_mask_query))
            score[i] = cos.cpu().detach().numpy()
        return torch.mean(score,dim=1)/torch.mean(self.weights)

        


    def predict(self, imgs_query, contour_querys):

        b = imgs_query.shape[0]
        losses = np.zeros(self.n_epochs,b)
        epochs_contours_query = np.zeros((self.n_epochs, b, contour_querys.shape[-2], 2))
        energies = np.zeros((self.n_epochs, b, len(self.activations), len(self.isolines)))
        stop = False
        scale = torch.tensor(
            [512.0, 512.0], device="cuda", dtype=torch.float32
        ) / torch.tensor(self.dims.copy(), device="cuda", dtype=torch.float32)

        imgs_array = np.transpose(imgs_query.cpu().detach().numpy(),[0,-1,1,2])
        imgs_array, _ = self.normalizer.normalize(imgs_array, stains=True)
        imgs_array = imgs_array / 255
        imgs_query = torch.from_numpy(imgs_array).permute(-1,0,1).unsqueeze(0)

        ### USE THE CLEANER
        contours_query_array = contours_query.cpu().detach().numpy()
        contours_query_array = contours_query_array / np.flip(self.dims)
        contours_query_array = self.cleaner.interpolate(contours_query_array, self.nb_points)
        contours_query = torch.from_numpy(np.roll(contours_query_array, axis=1, shift=1))
        contours_query.requires_grad = True

        #### pass image into neural network and get features
        _ = self.model(preprocess(imgs_query))


        for i in range(self.n_epochs):

            features_isoline_query = self.ctf(contours_query, self.activations)
            loss, energy = self.compute_multiscale_multiisoline_loss(features_isoline_query, weights)
            losses[i] = loss.cpu().detach().numpy()
            epochs_contours_query[i] = contours_query.cpu().detach().numpy()
            energies[i] = energy.cpu().detach().numpy()

            loss.backward(inputs=contours_query)

            norm_grad = torch.unsqueeze(torch.norm(contours_query.grad, dim=1), -1)
            clipped_norm = torch.clip(norm_grad, 0, self.clip)
            stop = torch.max(norm_grad) < self.thresh

            if stop == False:
                with torch.no_grad():
                    gradient_direction = contours_query.grad * clipped_norm / norm_grad
                    gradient_direction = self.smooth(gradient_direction)
                    contours_query = (
                        contours_query
                        - scale
                        * self.learning_rate
                        * (self.ed**i)
                        * gradient_direction
                    )   
                interpolated_contour = self.cleaner.clean_contours_and_interpolate(
                        contours_query.cpu().detach().numpy())
                contours_query = torch.from_numpy(interpolated_contour).cuda()
                contours_query.grad = None
                contours_query.requires_grad = True

            else:
                print("the algorithm stoped earlier")
                break


        ##### Calculate score after gradient descent
        _, features_mask_query = self.ctf(contours_query, self.activations)
        scores = self.compute_similarity_score(features_mask_query)
    
        contours_query = np.roll(contours_query, 1, -1)
        contours_query = (contours_query * np.flip(imgs_query.shape[-2:])).astype(np.int32)

        losses[losses == 0] = 1e10
        argmin = np.argmin(losses, axis = 0)

        return epochs_contours_query[argmin], scores, losses, energies
