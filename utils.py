import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import *
import numpy as np
import cv2
from torch.nn import Module
import torch
import torch.nn.functional as F
from torch_contour import *
from torch import cdist
from torchvision.transforms.functional import vflip, hflip


def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(
        x, [x < x0], [lambda x: k1 * x + y0 - k1 * x0, lambda x: k2 * x + y0 - k2 * x0]
    )


class Contour_to_features(torch.nn.Module):
    """
    A PyTorch neural network module designed to convert contour data into feature representations.
    This class leverages two sub-modules: Contour_to_mask and Mask_to_features.
    """

    def __init__(self, size: int, activations: dict):
        """
        Initializes the Contour_to_features class.

        This method creates instances of two sub-modules:
        - Contour_to_mask with a parameter of 200.
        - Mask_to_features with no parameters.


        Parameters:
        -----------
        activations: dict
            A Dictionary of feature maps (e.g., from a CNN).
            The dictionnary contains keys (int) which represent the order of the activations chosen by the user.
            Which means that activations[0] returns the 1st feature among the set of features in activations that have been chosen by the user.
            >>> Example 1: If the user wants to select each feature extracted by the model at each scale, activations(i) should contain the feature extracted at scale i.
            >>> Example 2: If the user wants to select each feature extracted by the model, activations(i) should contain the feature extracted at layer i.

        ctm : Contour_to_mask
            An instance of the Contour_to_mask class, initialized with a parameter of 200.
        mtf : Mask_to_features
            An instance of the Mask_to_features class.
        """
        super(Contour_to_features, self).__init__()
        self.ctm = Contour_to_mask(size, k=1e4).requires_grad_(False)
        self.mtf = Mask_to_features(activations).requires_grad_(False)

    def forward(self, contour):
        """
        Defines the forward pass of the Contour_to_features model.

        This method takes in a contour and activations, uses the Contour_to_mask sub-module to
        generate a mask from the contour, and then applies the Mask_to_features sub-module to
        combine the activations with the mask.

        Parameters:
        -----------
        contour : Tensor
            The contour data input tensor.

        Returns:
        --------
        output_features: List
            The output features after combining the activations with the mask.
            the output_features is a List.
            len(output_features) = len(activations)
            output_features[i] has shape (B, C, 1)

        """
        mask = self.ctm(contour)
        output_features = self.mtf(mask)
        return output_features


class Mask_to_features(Module):
    """
    A PyTorch neural network module designed to convert mask and activation data into feature representations.
    """

    def __init__(self, activations, eps=1e-5):
        """
        Initializes the Mask_to_features class.

        This method sets up the module without any specific parameters.

        Parameters:
        -----------
        activations: dict
            A Dictionary of feature maps (e.g., from a CNN).
            The dictionnary contains keys (int) which represent the order of the activations chosen by the user.
            Which means that activations[0] returns the 1st feature among the set of features in activations that have been chosen by the user.
            >>> Example 1: If the user wants to select each feature extracted by the model at each scale, activations(i) should contain the feature extracted at scale i.
            >>> Example 2: If the user wants to select each feature extracted by the model, activations(i) should contain the feature extracted at layer i.
        """
        super(Mask_to_features, self).__init__()
        self.activations = activations
        self.eps = eps

    def forward(self, mask: torch.Tensor):
        """
        Defines the forward pass of the Mask_to_features model.

        This method takes in a dictionary of activations and a mask tensor, resizes the mask to match the
        dimensions of each activation layer, and then calculates features inside and outside the mask for each
        activation layer.

        Parameters:
        -----------
        mask : torch.Tensor
            The mask tensor.

        Returns:
        --------
        features_inside : list of torch.Tensor
            A list containing the feature representations inside the mask for each activation layer.

        features_outside : list of torch.Tensor
            A list containing the feature representations outside the mask for each activation layer.

        len(features) = len(activations)
        features_outside[i] has shape (B, C_i, 1) and features_inside has shape (B, C_i, 1)
        with C_i the number of channels in self.activations[i]
        """

        masks = [
            F.interpolate(
                mask,
                size=(self.activations[i].shape[-2], self.activations[i].shape[-1]),
                mode="bilinear",
            )
            for i in range(len(self.activations))
        ]

        features_inside, features_outside = [], []

        for i in range(len(self.activations)):
            features_inside.append(
                (
                    torch.sum(self.activations[i] * masks[i], dim=(2, 3))
                    / (torch.sum(masks[i], (2, 3)) + self.eps)
                )[..., None]
            )

            features_outside.append(
                (
                    torch.sum(self.activations[i] * (1 - masks[i]), dim=(2, 3))
                    / (torch.sum((1 - masks[i]), (2, 3)) + self.eps)
                )[..., None]
            )

        return features_inside, features_outside


def augmentation(tuple_inputs_arrays):
    ps = np.random.random(10)
    for element in tuple_inputs_arrays:
        element = element.reshape((1, element.shape[0], element.shape[1], -1))

        if ps[0] > 1 / 4 and ps[0] < 1 / 2:
            element = torch.rot90(element, dims=(1, 2), k=1)

        if ps[0] > 1 / 2 and ps[0] < 3 / 4:
            element = torch.rot90(element, dims=(1, 2), k=2)

        if ps[0] > 3 / 4 and ps[0] < 1:
            element = torch.rot90(element, dims=(1, 2), k=3)

        if ps[1] > 0.5:
            element = vflip(element)

        if ps[2] > 0.5:
            element = hflip(element)

    return tuple_inputs_arrays


##### Change the doc
class Contour_to_isoline_features(torch.nn.Module):
    """
    A PyTorch neural network module designed to convert contour data into feature representations.
    This class leverages two sub-modules: Contour_to_mask and Mask_to_features.
    """

    def __init__(
        self,
        size: int,
        activations: dict,
        isolines: torch.Tensor,
        halfway_value: float,
        compute_features_mask=False,
    ):
        """
        Initializes the Contour_to_features class.

        This method creates instances of two sub-modules:
        - Contour_to_mask
        - Mask_to_features

        Parameters:
        -----------
        size: int
            the size of image containing the normalized distance map generated in order to retrieve the isolines_features
        activations: dict
            A Dictionary of feature maps (e.g., from a CNN).
            The dictionnary contains keys (int) which represent the order of the activations chosen by the user.
            Which means that activations[0] returns the 1st feature among the set of features in activations that have been chosen by the user.
            >>> Example 1: If the user wants to select each feature extracted by the model at each scale, activations(i) should contain the feature extracted at scale i.
            >>> Example 2: If the user wants to select each feature extracted by the model, activations(i) should contain the feature extracted at layer i.
        ctd : Contour_to_mask
            An instance of the Contour_to_distance_map class.
        dtf : Distance_map_to_features
            An instance of the istance_map_to_features class.
        compute_features_mask : bool
            whether to compute the average features at each scale inside the mask or not

        """
        super(Contour_to_isoline_features, self).__init__()
        self.ctd = Contour_to_distance_map(size).requires_grad_(False)
        self.dtf = Distance_map_to_isoline_features(
            activations, isolines, halfway_value
        ).requires_grad_(False)
        self.compute_features_mask = compute_features_mask

    def forward(self, contour):
        """
        Defines the forward pass of the Contour_to_features model.

        This method takes in a contour and activations, uses the Contour_to_mask sub-module to
        generate a mask from the contour, and then applies the Mask_to_features sub-module to
        combine the activations with the mask.

        Parameters:
        -----------
        contour : Tensor
            The contour data input tensor.
        Returns:
        --------
        output_features: tuple of list of tensors
            The output features after combining the activations with the mask.
            if self.compute_features_mask = True, the output_features will be a tuple.
                output_features[0] correspond to the list of the features at each scale  and each isoline.
                output_features[1] correspond to the list of the features inside the mask at each scale.
            if self.compute_features_mask = False
                output_features correspond to the list of the features at each scale  and each isoline.


        """
        self.dtf.compute_features_mask = self.compute_features_mask
        dmap, mask = self.ctd(contour, True)
        output_features = self.dtf(dmap, mask)
        return output_features


class Distance_map_to_isoline_features(Module):
    def __init__(
        self,
        activations: dict,
        isolines: torch.Tensor,
        halfway_value: float = 0.5,
        compute_features_mask=False,
    ):
        """
        Initializes the Isoline_to_features class.

        Parameters:
        -----------
        activations: dict
            A Dictionary of feature maps (e.g., from a CNN).
            The dictionnary contains keys (int) which represent the order of the activations chosen by the user.
            Which means that activations[0] returns the 1st feature among the set of features in activations that have been chosen by the user.
            >>> Example 1: If the user wants to select each feature extracted by the model at each scale, activations(i) should contain the feature extracted at scale i.
            >>> Example 2: If the user wants to select each feature extracted by the model, activations(i) should contain the feature extracted at layer i.

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
        self.vars = self.mean_to_var(
            self.isolines, halfway_value
        )  # Store the variance tensor.
        self.activations = activations
        self.compute_features_mask = compute_features_mask

    def mean_to_var(self, isolines, halfway_value):
        """

        This function takes a list of isolines values (which correspond to the mean values of the gaussians)
        and computes the variances of each gaussian so that two consecutive gaussians sum to halfway_value at halfway the means.

        Parameters:
        -----------
        isolines: torch.Tensor
            A tensor representing isoline values.
        halfway_value: float
        The value that must be reached in the middle of two consecutive isolines (represented as gaussians) when summing them together.

        Returns:
        --------
        variances: torch.Tensor
            The variances of each gaussian so that two consecutive isolines sum to halfway_value at halfway.
            len(variances) = len(isolines)
        """

        mat = cdist(isolines[:, None], isolines[:, None]) ** 2
        mat = torch.where(mat == 0, torch.tensor(float("inf")), mat)
        variances = -torch.min(mat, 0).values / (8 * np.log(halfway_value))
        return variances

    def forward(self, distance_map: torch.Tensor, mask: torch.Tensor):
        """
        Forward pass of the Isoline_to_features module. Generates features from activations and isolines.

        Parameters:
        -----------

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
        nb_scales = len(self.activations)

        # Apply Gaussian-like weighting to isolines based on distance_map and variance
        isolines = mask * torch.exp(
            -((self.isolines[None, :, None, None] - distance_map) ** (2))
            / (self.vars[None, :, None, None])
        )

        # Resize the isolines to match each activation scale, using bilinear interpolation
        isolines_scales = [
            F.interpolate(
                isolines,
                size=(
                    self.activations[i].shape[-2],
                    self.activations[i].shape[-1],
                ),  # Match activations' spatial size
                mode="bilinear",
            )
            for i in range(nb_scales)
        ]

        # If compute_features_mask is True, resize the mask for each scale
        if self.compute_features_mask:
            masks = [
                F.interpolate(
                    mask,
                    size=(
                        self.activations[i].shape[-2],
                        self.activations[i].shape[-1],
                    ),  # Match activations' spatial size
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
            f_s_i = (self.activations[i][:, :, None] * isolines_scales[i][:, None]).sum(
                dim=[-2, -1]
            ) / isolines_scales[i].sum(dim=[-2, -1])[:, None]
            features_isolines.append(f_s_i)

            # If compute_features_mask is True, compute and store features based on masks
            if self.compute_features_mask:
                features_mask.append(
                    torch.sum(
                        self.activations[i] * masks[i], dim=(-2, -1)
                    )  # Compute masked feature aggregation
                    / torch.sum(masks[i], dim=(-2, -1))  # Normalize by mask's sum
                )

        # Return the features and features_mask (if computed)
        return features_isolines, features_mask


def define_contour_init(n, center, axes, angle=0):
    # major, minor axes
    start_angle = 0
    end_angle = 360
    color = 1
    thickness = -1

    # Draw a filled ellipse on the input image
    mask = cv2.ellipse(
        np.zeros((n, n)),
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
