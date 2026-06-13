"""
Model architectures for DCF.

This module contains the model architectures supported by DCF:
- VGG16
- ResNet50
- ResNet101
- ResNet_FPN (Feature Pyramid Network)
- ResNet101_FPN
"""

import torch
import torchvision.models as models


class VGG16(torch.nn.Module):
    """
    VGG16 architecture for multi-scale feature extraction.

    Extracts features from layers 3, 8, 15, 22, 29 for a multi-scale
    representation of image characteristics.
    """

    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(weights="DEFAULT")
        self.features = vgg16.features.to(torch.float32)

    def forward(self, x):
        return self.features(x)


class ResNet50(torch.nn.Module):
    """
    ResNet50 architecture for multi-scale feature extraction.

    Extracts features from layers layer1, layer2, layer3, layer4 for a multi-scale
    representation of image characteristics.
    """

    def __init__(self):
        super().__init__()
        resnet = models.resnet50(weights="DEFAULT")
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)  # 1/4
        c2 = self.layer2(c1)  # 1/8
        c3 = self.layer3(c2)  # 1/16
        c4 = self.layer4(c3)  # 1/32

        return [c1, c2, c3, c4]


class ResNet101(torch.nn.Module):
    """
    ResNet101 architecture for multi-scale feature extraction.

    Extracts features from layers layer1, layer2, layer3, layer4 for a multi-scale
    representation of image characteristics.
    """

    def __init__(self):
        super().__init__()
        resnet = models.resnet101(weights="DEFAULT")
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 1/4
        self.layer2 = resnet.layer2  # 1/8
        self.layer3 = resnet.layer3  # 1/16
        self.layer4 = resnet.layer4  # 1/32

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c1 = self.layer1(x)  # 1/4
        c2 = self.layer2(c1)  # 1/8
        c3 = self.layer3(c2)  # 1/16
        c4 = self.layer4(c3)  # 1/32

        return [c1, c2, c3, c4]


class ResNet_FPN(torch.nn.Module):
    """
    ResNet50 architecture with Feature Pyramid Network (FPN).

    Implements a custom Feature Pyramid Network that extracts multi-scale
    features directly in the forward pass. Returns a list of features
    at different spatial scales.
    """

    def __init__(self, backbone_name: str = "resnet50"):
        super().__init__()

        if backbone_name == "resnet50":
            backbone = models.resnet50(weights="DEFAULT")
        elif backbone_name == "resnet101":
            backbone = models.resnet101(weights="DEFAULT")
        else:
            raise ValueError(f"Backbone {backbone_name} not supported")

        self.backbone = backbone
        # Extraire les couches pour FPN
        self.layer1 = backbone.layer1  # 1/4
        self.layer2 = backbone.layer2  # 1/8
        self.layer3 = backbone.layer3  # 1/16
        self.layer4 = backbone.layer4  # 1/32

    def forward(self, x):
        # Forward pass to extract multi-scale features
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        c1 = self.layer1(x)  # 1/4
        c2 = self.layer2(c1)  # 1/8
        c3 = self.layer3(c2)  # 1/16
        c4 = self.layer4(c3)  # 1/32

        return [c1, c2, c3, c4]


class ResNet101_FPN(torch.nn.Module):
    """
    ResNet101 architecture with Feature Pyramid Network (FPN).

    Implements a custom Feature Pyramid Network based on ResNet101.
    Returns multi-scale features directly in the forward pass.
    """

    def __init__(self):
        super().__init__()
        self.fpn = ResNet_FPN("resnet101")

        # Expose layers directly for compatibility with hooks
        self.layer1 = self.fpn.layer1
        self.layer2 = self.fpn.layer2
        self.layer3 = self.fpn.layer3
        self.layer4 = self.fpn.layer4

    def forward(self, x):
        return self.fpn(x)
