"""
Module de configuration pour les modèles supportés par DCF.

Ce module fournit une interface unifiée pour charger et configurer différents modèles
de deep learning (VGG16, ResNet50, ResNet101, ResNet-FPN) pour l'extraction de features.
"""

import logging
from typing import Any, Dict

import torch
from torchvision import transforms

# Import des architectures de modèles
from models.models_architectures import (
    VGG16,
    ResNet50,
    ResNet101,
    ResNet101_FPN,
    ResNet_FPN,
)

logger = logging.getLogger(__name__)

preprocess = transforms.Compose(
    [
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

MODEL_CONFIGS = {
    "vgg16": {
        "model_fn": lambda: VGG16(),
        "layer_indices": [3, 8, 15, 22, 29],
        "layer_access": lambda model, idx: model.features[idx]
        if hasattr(model.features, str(idx))
        else None,
        "preprocess": preprocess,
        "description": "VGG16 with features extracted from layers 3, 8, 15, 22, 29",
    },
    "resnet50": {
        "model_fn": lambda: ResNet50(),
        "layer_indices": ["layer1", "layer2", "layer3", "layer4"],
        "layer_access": lambda model, idx: getattr(model, idx)
        if hasattr(model, idx)
        else None,
        "preprocess": preprocess,
        "description": "ResNet50 with features extracted from layers layer1-4",
    },
    "resnet101": {
        "model_fn": lambda: ResNet101(),
        "layer_indices": ["layer1", "layer2", "layer3", "layer4"],
        "layer_access": lambda model, idx: getattr(model, idx)
        if hasattr(model, idx)
        else None,
        "preprocess": preprocess,
        "description": "ResNet101 with features extracted from layers layer1-4",
    },
    "resnet_fpn": {
        "model_fn": lambda: ResNet_FPN("resnet50"),
        "layer_indices": ["layer1", "layer2", "layer3", "layer4"],
        "layer_access": lambda model, idx: getattr(model, idx)
        if hasattr(model, idx)
        else None,
        "preprocess": preprocess,
        "description": "ResNet50 with Feature Pyramid Network (FPN)",
    },
    "resnet101_fpn": {
        "model_fn": lambda: ResNet101_FPN(),
        "layer_indices": ["layer1", "layer2", "layer3", "layer4"],
        "layer_access": lambda model, idx: getattr(model, idx)
        if hasattr(model, idx)
        else None,
        "preprocess": preprocess,
        "description": "ResNet101 with Feature Pyramid Network (FPN)",
    },
}


def create_resnet_fpn_model(backbone_name: str = "resnet50") -> torch.nn.Module:
    """
    Creates a ResNet model with Feature Pyramid Network (FPN).

    Args:
        backbone_name: Backbone name ('resnet50' or 'resnet101')

    Returns:
        ResNet model with FPN

    Raises:
        ValueError: If the backbone is not supported
    """
    try:
        if backbone_name == "resnet50":
            return ResNet_FPN("resnet50")
        elif backbone_name == "resnet101":
            return ResNet_FPN("resnet101")
        else:
            raise ValueError(f"Backbone {backbone_name} not supported")

    except Exception as e:
        logger.error(f"Error creating ResNet-FPN model: {e}")
        raise


def detect_model_type(model: torch.nn.Module) -> str:
    """
    Automatically detects the model type.

    Args:
        model: PyTorch model

    Returns:
        Detected model type ('vgg16', 'resnet50', 'resnet101', 'resnet_fpn')
    """
    model_str = str(model)
    model_class_name = model.__class__.__name__

    if "ResNet" in model_str:
        if model_class_name == "ResNet50":
            return "resnet50"
        elif model_class_name == "ResNet101":
            return "resnet101"
        elif model_class_name == "ResNet_FPN":
            return "resnet_fpn"
        elif model_class_name == "ResNet101_FPN":
            return "resnet101_fpn"
        elif "FPN" in model_str:
            return "resnet_fpn"
        else:
            # Fallback based on class name
            if "resnet50" in model_str.lower():
                return "resnet50"
            elif "resnet101" in model_str.lower():
                return "resnet101"
            else:
                return "resnet50"  # Default
    elif "VGG" in model_str or "Sequential" in model_str or model_class_name == "VGG16":
        return "vgg16"
    else:
        logger.warning(f"Unrecognized model type: {model_str}. Using VGG16 as default.")
        return "vgg16"


def get_model_config(model_type: str) -> Dict[str, Any]:
    """
    Retrieves the configuration of a specific model.

    Args:
        model_type: Model type ('vgg16', 'resnet50', 'resnet101', 'resnet_fpn', 'resnet101_fpn')

    Returns:
        Model configuration

    Raises:
        ValueError: If the model type is not supported
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model type: {model_type}")

    return MODEL_CONFIGS[model_type]


def list_available_models() -> Dict[str, str]:
    """
    Lists all available models with their descriptions.

    Returns:
        Dictionary of available models with their descriptions
    """
    return {name: config["description"] for name, config in MODEL_CONFIGS.items()}


def create_model(model_type: str) -> torch.nn.Module:
    """
    Creates a model of the specified type.

    Args:
        model_type: Model type to create

    Returns:
        Initialized PyTorch model

    Raises:
        ValueError: If the model type is not supported
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model type: {model_type}")

    try:
        return MODEL_CONFIGS[model_type]["model_fn"]()
    except Exception as e:
        logger.error(f"Error creating model {model_type}: {e}")
        raise


def get_model_layer_access(model_type: str):
    """
    Retrieves the layer access function for a model type.

    Args:
        model_type: Model type

    Returns:
        Layer access function
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model type: {model_type}")

    return MODEL_CONFIGS[model_type]["layer_access"]


def get_model_layer_indices(model_type: str) -> list:
    """
    Retrieves the layer indices for a model type.

    Args:
        model_type: Model type

    Returns:
        List of layer indices
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model type: {model_type}")

    return MODEL_CONFIGS[model_type]["layer_indices"]


def get_model_preprocess(model_type: str):
    """
    Retrieves the preprocessing function for a model type.

    Args:
        model_type: Model type

    Returns:
        Preprocessing function
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model type: {model_type}")

    return MODEL_CONFIGS[model_type]["preprocess"]
