import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import *
import numpy as np
import cv2
from torch.nn import Module
import torch
import torch.nn.functional as F
from torch_contour.torch_contour import Contour_to_mask, Contour_to_distance_map
from torch import cdist


def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])
class Contour_to_features(torch.nn.Module):
    """
    A PyTorch neural network module designed to convert contour data into feature representations.
    This class leverages two sub-modules: Contour_to_mask and Mask_to_features.
    """

    def __init__(self, size:int):
        """
        Initializes the Contour_to_features class.

        This method creates instances of two sub-modules:
        - Contour_to_mask with a parameter of 200.
        - Mask_to_features with no parameters.

        Attributes:
        -----------
        ctm : Contour_to_mask
            An instance of the Contour_to_mask class, initialized with a parameter of 200.
        mtf : Mask_to_features
            An instance of the Mask_to_features class.
        """
        super(Contour_to_features, self).__init__()
        self.ctm = Contour_to_mask(size,k=1e4).requires_grad_(False)
        self.mtf = Mask_to_features().requires_grad_(False)


    def forward(self, contour, activations):
        """
        Defines the forward pass of the Contour_to_features model.

        This method takes in a contour and activations, uses the Contour_to_mask sub-module to
        generate a mask from the contour, and then applies the Mask_to_features sub-module to
        combine the activations with the mask.

        Parameters:
        -----------
        contour : Tensor
            The contour data input tensor.
        activations : Tensor
            The activations input tensor.

        Returns:
        --------
        Tensor
            The output features after combining the activations with the mask.
        """
        mask = self.ctm(contour)
        return self.mtf(activations, mask)


class Mask_to_features(Module):
    """
    A PyTorch neural network module designed to convert mask and activation data into feature representations.
    """

    def __init__(self):
        """
        Initializes the Mask_to_features class.

        This method sets up the module without any specific parameters.
        """
        super(Mask_to_features, self).__init__()

    def forward(self, activations: dict, mask: torch.Tensor, eps=1e-3):
        """
        Defines the forward pass of the Mask_to_features model.

        This method takes in a dictionary of activations and a mask tensor, resizes the mask to match the
        dimensions of each activation layer, and then calculates features inside and outside the mask for each
        activation layer.

        Parameters:
        -----------
        activations: dict
            A Dictionary of feature maps (e.g., from a CNN).
            The dictionnary contains keys (int) which represent the order of the activations chosen by the user.
            Which means that activations[0] returns the 1st feature among the set of features in activations that have been chosen by the user.
            >>> Example 1: If the user wants to select each feature extracted by the model at each scale, activations(i) should contain the feature extracted at scale i.
            >>> Example 2: If the user wants to select each feature extracted by the model, activations(i) should contain the feature extracted at layer i.

        mask : torch.Tensor
            The mask tensor.

        Returns:
        --------
        features_inside : list of torch.Tensor
            A list containing the feature representations inside the mask for each activation layer.
        features_outside : list of torch.Tensor
            A list containing the feature representations outside the mask for each activation layer.
        """

        masks = [
            F.interpolate(
                mask,
                size=(activations[i].shape[-2], activations[i].shape[-1]),
                mode="bilinear",
            )
            for i in range(len(activations))
        ]

        features_inside, features_outside = [], []

        for i in range(len(activations)):

            features_inside.append(
                torch.sum(activations[i] * masks[i], dim=(2, 3))
                / (torch.sum(masks[i], (2, 3)) + eps)
            )

            features_outside.append(
                torch.sum(activations[i] * (1 - masks[i]), dim=(2, 3))
                / (torch.sum((1 - masks[i]), (2, 3)) + eps)
            )

        return features_inside, features_outside


def augmentation(img, mask):
    img = img.reshape((-1, img.shape[0], img.shape[1],3))
    mask = mask.reshape((1, mask.shape[0], mask.shape[1],-1))

    ps = np.random.random(10)

    if ps[0] > 1 / 4 and ps[0] < 1 / 2:
        img, mask = np.rot90(img, axes=[1, 2], k=1), np.rot90(mask, axes=[1, 2], k=1)

    if ps[0] > 1 / 2 and ps[0] < 3 / 4:
        img, mask = np.rot90(img, axes=[1, 2], k=2), np.rot90(mask, axes=[1, 2], k=2)

    if ps[0] > 3 / 4 and ps[0] < 1:
        img, mask = np.rot90(img, axes=[1, 2], k=3), np.rot90(mask, axes=[1, 2], k=3)

    if ps[1] > 0.5:
        img, mask = np.flip(img, 1), np.flip(mask, 1)

    if ps[2] > 0.5:
        img, mask = np.flip(img, 2), np.flip(mask, 2)

    return img, mask


##### Change the doc
class Contour_to_isoline_features(torch.nn.Module):
    """
    A PyTorch neural network module designed to convert contour data into feature representations.
    This class leverages two sub-modules: Contour_to_mask and Mask_to_features.
    """

    def __init__(self, size: int, isolines: torch.Tensor, halfway_value: float, compute_features_mask=False):
        """
        Initializes the Contour_to_features class.

        This method creates instances of two sub-modules:
        - Contour_to_mask with a parameter of 200.
        - Mask_to_features with no parameters.

        Attributes:
        -----------
        ctm : Contour_to_mask
            An instance of the Contour_to_mask class, initialized with a parameter of 200.
        mtf : Mask_to_features
            An instance of the Mask_to_features class.
        """
        super(Contour_to_isoline_features, self).__init__()
        self.ctd = Contour_to_distance_map(size).requires_grad_(False)
        self.dtf = Distance_map_to_isoline_features(isolines, halfway_value).requires_grad_(False)
        self.compute_features_mask = compute_features_mask

    def forward(self, contour, activations):
        """
        Defines the forward pass of the Contour_to_features model.

        This method takes in a contour and activations, uses the Contour_to_mask sub-module to
        generate a mask from the contour, and then applies the Mask_to_features sub-module to
        combine the activations with the mask.

        Parameters:
        -----------
        contour : Tensor
            The contour data input tensor.
        activations : Tensor
            The activations input tensor.

        Returns:
        --------
        Tensor
            The output features after combining the activations with the mask.
        """
        dmap, mask = self.ctd(contour, True)
        return self.dtf(activations, dmap, mask, compute_features_mask=self.compute_features_mask)
    

class Distance_map_to_isoline_features(Module):
    def __init__(self, isolines: torch.Tensor, halfway_value: float=0.5):

        """
        Initializes the Isoline_to_features class.

        Parameters:
        -----------
        isolines: torch.Tensor
            A tensor representing isoline values.
            the isolines values must in [0, 1]
            Example: torch.tensor([0.0, 0.5, 0.8])
            
        halfway_value: float
        halfway_value is the value that must be reached in the middle of two consecutive isolines (represented as gaussians) when summing them together.
        
        >>> For example if isolines = [0,1] 
        >>> and halfway value = 0.8 
        >>> then the variances of the gaussian centered on 0 and the gaussian centered on 1 should be set so that thety sum up to 0.8 at 0.5.
            
        
        """
        super(Distance_map_to_isoline_features, self).__init__()

        self.isolines = isolines  # Store the isoline tensor.
        self.vars = self.mean_to_var(self.isolines, halfway_value)  # Store the variance tensor.

    def mean_to_var(self, isolines, halfway_value):

        mat = cdist(isolines[:, None], isolines[:, None]) ** 2
        mat = torch.where(mat == 0, torch.tensor(float('inf')), mat)
        masked_a = -torch.min(mat, 0) / (
            8 * torch.log(halfway_value)
        )
        return masked_a
    
    def forward(
        self,
        activations: dict,
        distance_map: torch.Tensor,
        mask: torch.Tensor,
        compute_features_mask=False,
    ):
        """
        Forward pass of the Isoline_to_features module. Generates features from activations and isolines.

        Parameters:
        -----------
        activations: dict
            A Dictionary of multi-scale feature maps (e.g., from a CNN).
            The dictionnary contains keys (int) which represent the order of the activations cho by the user.
            Which means that activations[0] returns the 1st feature among the set of features in activations that have been chosen by the user.
            If the user wants to select each feature extracted by the model at each scale, activations(i) should contain the feature extracted at scale i.
        
        distance_map: torch.Tensor
            A tensor with shape (B, 1, H, W)
            The tensor represents a batch of distance maps.
        
        mask: torch.Tensor:
            A tensor with shape (B, 1, H, W)
            the mask of each contour  in the batch.

        compute_features_mask: (bool, optional)
            If True, compute additional aggregated features inside the masks for each features in activations.

        Returns:
        --------

        features_isolines: list
            A list of aggregated features at each isoline for each feature in activations.
            
        features_mask:list
            A list of aggregated features inside the mask for each features in activations (if compute_features_mask is True).
        """
        
        # Number of scales in the activations dictionary
        nb_scales = len(activations)
        
        # Apply Gaussian-like weighting to isolines based on distance_map and variance
        isolines = mask * torch.exp(
            -((self.isolines[None, :, None, None] - distance_map) ** (2)) 
            / (self.vars[None,:, None, None])
        )

        # Resize the isolines to match each activation scale, using bilinear interpolation
        isolines_scales = [
            F.interpolate(
                isolines,
                size=(activations[i].shape[-2], activations[i].shape[-1]),  # Match activations' spatial size
                mode="bilinear",
            )
            for i in range(nb_scales)
        ]

        # If compute_features_mask is True, resize the mask for each scale
        if compute_features_mask:
            masks = [
                F.interpolate(
                    mask,  
                    size=(activations[i].shape[-2], activations[i].shape[-1]),  # Match activations' spatial size
                    mode="bilinear",
                )
                for i in range(nb_scales)
            ]

        # Initialize lists for features and features_mask (if applicable)
        features_isolines, features_mask = [], []

        # Loop through each scale and compute features
        for i in range(nb_scales):


            # Compute feature aggregation at scale 'i' by summing over the spatial dimensions,
            # weighted by the isolines, and normalizing by the sum of isolines
            f_s_i = (activations[i][:,:,None] * isolines_scales[i][:,None]).sum(dim=[-2,-1]) / isolines_scales[i].sum(dim=[-2,-1])[..., None]
            features_isolines.append(f_s_i)

            # If compute_features_mask is True, compute and store features based on masks
            if compute_features_mask:
                features_mask.append(
                    torch.sum(activations[i] * masks[i], dim=(-2,-1))  # Compute masked feature aggregation
                    / torch.sum(masks[i], dim=(-2,-1))  # Normalize by mask's sum
                )
        
        # Return the features and features_mask (if computed)
        return features_isolines, features_mask



def define_contour_init(img, center, axes, angle=0):
    # major, minor axes
    start_angle = 0
    end_angle = 360
    color = 1
    thickness = -1

    # Draw a filled ellipse on the input image
    mask = cv2.ellipse(
        np.zeros(img.shape[:-1]),
        center,
        axes,
        angle,
        start_angle,
        end_angle,
        color,
        thickness,
    ).astype(np.uint8)
    contour = np.squeeze(
        cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0][0]
    )
    return contour, mask
