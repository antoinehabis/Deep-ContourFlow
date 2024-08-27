import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import *
import numpy as np
import cv2
from torch.nn import Module
import torch
import torch.nn.functional as F
from torch_contour.torch_contour import Contour_to_mask
import matplotlib.pyplot as plt


class Contour_to_features(torch.nn.Module):
    """
    A PyTorch neural network module designed to convert contour data into feature representations.
    This class leverages two sub-modules: Contour_to_mask and Mask_to_features.
    """

    def __init__(self, size):
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
        self.ctm = Contour_to_mask(size).requires_grad_(False)
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

    def forward(self, activations: dict, mask: torch.Tensor):
        """
        Defines the forward pass of the Mask_to_features model.

        This method takes in a dictionary of activations and a mask tensor, resizes the mask to match the
        dimensions of each activation layer, and then calculates features inside and outside the mask for each
        activation layer.

        Parameters:
        -----------
        activations : dict
            A dictionary containing activation tensors with keys as layer indices.
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
                size=(activations[str(i)].shape[-2], activations[str(i)].shape[-1]),
                mode="bilinear",
            )
            for i in range(5)
        ]

        features_inside, features_outside = [], []

        for i in range(len(activations)):

            features_inside.append(
                torch.sum(activations[str(i)] * masks[i], dim=(2, 3))
                / (torch.sum(masks[i], (2, 3)) + 1e-5)
            )

            features_outside.append(
                torch.sum(activations[str(i)] * (1 - masks[i]), dim=(2, 3))
                / (torch.sum((1 - masks[i]), (2, 3)) + 1e-5)
            )

        return features_inside, features_outside


def augmentation(img, mask):
    img = (img * 255).astype(np.uint8)
    mask_shape = mask.shape

    if len(mask_shape) == 2:
        mask = np.expand_dims(mask, -1)

    ps = np.random.random(10)
    if ps[0] > 1 / 4 and ps[0] < 1 / 2:
        img, mask = np.rot90(img, axes=[0, 1], k=1), np.rot90(mask, axes=[0, 1], k=1)

    if ps[0] > 1 / 2 and ps[0] < 3 / 4:
        img, mask = np.rot90(img, axes=[0, 1], k=2), np.rot90(mask, axes=[0, 1], k=2)

    if ps[0] > 3 / 4 and ps[0] < 1:
        img, mask = np.rot90(img, axes=[0, 1], k=3), np.rot90(mask, axes=[0, 1], k=3)

    if ps[1] > 0.5:
        img, mask = np.flipud(img), np.flipud(mask)

    if ps[2] > 0.5:
        img, mask = np.fliplr(img), np.fliplr(mask)
    mask = mask[:, :]
    img = img / 255
    return img, mask


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
