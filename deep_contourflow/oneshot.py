import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, label
from torch_contour import CleanContours, Contour_to_distance_map, Contour_to_mask, Smoothing, area
from torchvision import models, transforms
from tqdm import tqdm

from .features import (
    AVAILABLE_AUGMENTATIONS,
    Contour_to_features,
    Contour_to_isoline_features,
    Distance_map_to_isoline_features,
    Mask_to_features,
    augmentation,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

preprocess = transforms.Compose(
    [
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

try:
    vgg16 = models.vgg16(weights="DEFAULT")
    VGG16 = vgg16.features.to(torch.float32)
except Exception as e:
    logger.error(f"Error loading VGG16 model: {e}")
    raise


class DCF:
    def __init__(
        self,
        n_epochs: int = 100,
        nb_augment: int = 100,
        model: torch.nn.Module = VGG16,
        sigma: float = 7,
        learning_rate: float = 5e-2,
        clip: float = 1e-1,
        exponential_decay: float = 0.998,
        thresh: float = 1e-2,
        isolines: Optional[List[float]] = None,
        isoline_weights: Optional[List[float]] = None,
        lambda_area: float = 1e-4,
        augmentations: Optional[List[str]] = None,
        early_stopping_patience: int = 10,
        early_stopping_threshold: float = 1e-6,
        use_mixed_precision: bool = False,
        device: Optional[str] = None,
        do_apply_grabcut: bool = False,
    ):
        """
        This class implements the one shot version of DCF. It contains a fit and a predict step.
        The fit step aims at capturing the features of the support image in the support contour.
        The predict step aims at evolving an initial contour so that the features match as much as possible to the ones of the support

        Parameters:
        -----------
        n_epochs : int
            The maximum number of gradient descent during the predict step.

        nb_augment : int
            The number of augmentations applied to the support image during the fitting step.
            Note that if you want to apply your own augmentations please go to utils.py and modify the augmentation method.

        model: torch.nn.Module
            The pretrained model from which the activations will be extracted.
            This model can be any model as long as the activations have shape (B,C,H,W).
            Note that for each model you choose to work with you will have to specify which activations of the model you want to use.
            For example if you are interested in the activations from the 3rd, 8th, 15th, 22th, 29th layers of the model VGG16 please write

            >>> self.model[3].register_forward_hook(self.get_activations("0"))
            >>> self.model[8].register_forward_hook(self.get_activations("1"))
            >>> self.model[15].register_forward_hook(self.get_activations("2"))
            >>> self.model[22].register_forward_hook(self.get_activations("3"))
            >>> self.model[29].register_forward_hook(self.get_activations("4"))

            You don't have to use 5 layers but we do in the paper.

        sigma: float
            The standard deviation of the gaussian smoothing operator.

        learning_rate: float
            The value of the gradient step.

        clip: float
            The value to set in order to clip the norm of the gradient of the contour so that it doesn't move too far.

        exponential_decay: float
            The exponential decay of the learning_rate.

        thresh: float
            If the maximum of the norm of the gradient of the contour over each node does not exceed thresh, we stop the contour evolution.

        isolines: List[float]
            Values in the list must be in the range [0,1]
            If provided, DCF will use the isolines centered on the values inside the list and use them to move the contour over time.
            If None, DCF won't use any isoline and will move the contour using the aggregation of the features inside the mask corresponding to the contour.

        isoline_weights: List[float]
            The corresponding weights values w_i for each isoline in isolines when computing the loss.

        lambda_area: float
            Weight for the area constraint in the loss function.

        augmentations: List[str]
            Subset of augmentations to apply during fitting. Each element must be one of
            ["rot90", "vflip", "hflip"]. None (default) applies all three.

        early_stopping_patience: int
            Number of epochs to wait before stopping if loss doesn't improve.

        early_stopping_threshold: float
            Minimum improvement threshold for early stopping.

        use_mixed_precision: bool
            Whether to use mixed precision training for faster computation.

        device: Optional[str]
            Device to use for computation. If None, will be automatically detected.
        """

        self._validate_parameters(
            n_epochs,
            nb_augment,
            sigma,
            learning_rate,
            clip,
            exponential_decay,
            thresh,
            lambda_area,
            augmentations,
            early_stopping_patience,
            early_stopping_threshold,
        )

        self.n_epochs = n_epochs
        self.nb_augment = nb_augment
        self.model = model
        self.learning_rate = learning_rate
        self.clip = clip
        self.ed = exponential_decay
        self.thresh = thresh
        self.lambda_area = lambda_area
        self.augmentations = augmentations  # None → all; validated in _validate_parameters
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.use_mixed_precision = use_mixed_precision
        self.do_apply_grabcut = do_apply_grabcut

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Mixed precision setup
        if self.use_mixed_precision and self.device.type == "cuda":
            self.scaler = torch.amp.GradScaler('cuda')
        else:
            self.scaler = None

        # Enable optimizations
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

        self._initialize_components(sigma, isolines, isoline_weights)

        logger.info(
            f"DCF initialized with {n_epochs} epochs, lr={learning_rate}, device={self.device}"
        )

    def _validate_parameters(
        self,
        n_epochs: int,
        nb_augment: int,
        sigma: float,
        learning_rate: float,
        clip: float,
        exponential_decay: float,
        thresh: float,
        lambda_area: float,
        augmentations,
        early_stopping_patience: int,
        early_stopping_threshold: float,
    ) -> None:
        """Validate input parameters."""
        if n_epochs <= 0:
            raise ValueError("n_epochs must be positive")
        if nb_augment <= 0:
            raise ValueError("nb_augment must be positive")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if clip <= 0:
            raise ValueError("clip must be positive")
        if not 0 < exponential_decay < 1:
            raise ValueError("exponential_decay must be between 0 and 1")
        if thresh <= 0:
            raise ValueError("thresh must be positive")
        if lambda_area < 0:
            raise ValueError("lambda_area must be non-negative")
        if augmentations is not None:
            unknown = set(augmentations) - set(AVAILABLE_AUGMENTATIONS)
            if unknown:
                raise ValueError(
                    f"Unknown augmentation(s): {unknown}. Choose from {AVAILABLE_AUGMENTATIONS}."
                )
        if early_stopping_patience <= 0:
            raise ValueError("early_stopping_patience must be positive")
        if early_stopping_threshold <= 0:
            raise ValueError("early_stopping_threshold must be positive")

    def _initialize_components(self, sigma: float, isolines, isoline_weights) -> None:
        """Initialize algorithm components."""
        try:
            self.activations = {}
            self.isolines = (
                torch.tensor(isolines, dtype=torch.float32)
                if isolines is not None
                else None
            )
            self.isoline_weights = (
                torch.tensor(isoline_weights, dtype=torch.float32)
                if isoline_weights is not None
                else None
            )

            self._setup_activation_hooks()

            self.smooth = Smoothing(sigma)
            self.cleaner = CleanContours()

        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise

    def _setup_activation_hooks(self) -> None:
        """Configure hooks for extracting model activations."""
        try:
            # VGG16 layers for multi-scale feature extraction
            layer_indices = [3, 8, 15, 22, 29]
            for i, layer_idx in enumerate(layer_indices):
                if layer_idx < len(self.model):
                    self.model[layer_idx].register_forward_hook(self.get_activations(i))
                else:
                    logger.warning(f"Layer {layer_idx} not found in model")
        except Exception as e:
            logger.error(f"Error configuring hooks: {e}")
            raise

    def get_activations(self, name: int):
        """
        Returns a hook function that stores the activations (outputs) of a layer in a dictionary
        under the given name. This hook is designed to be registered on a specific layer in a model,
        allowing you to capture its output (activations) during the forward pass.

        Parameters:
        -----------
        name : int
             An integer that identifies the name/key under which the activations of the layer
             should be stored in `self.activations`.

        Returns:
        --------
        hook : function
            A hook function that takes the model, input, and output as arguments. It captures
            the output (activations) of the layer and stores it in the `self.activations`
            dictionary, ensuring the tensor is moved to the correct device.
        """

        def hook(model, input, output):
            """Hook function that captures the activations of the layer and stores them.

            Parameters:
            -----------
            model : torch.nn.Module
                  The layer from which activations are being captured.
            input : torch.Tensor
                  Input to the layer. It's used here to get the device information.
            output : torch.Tensor
                   The output (activations) of the layer, which will be stored in `self.activations`.

            Returns:
            --------
            None
            """
            try:
                device = input[0].device
                self.activations[name] = output.to(device)
            except Exception as e:
                logger.error(f"Error capturing activations: {e}")
                raise

        return hook

    def multi_scale_multi_isoline_loss(
        self, features_isolines: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the multi-scale multi-isoline loss between the query features and support features across multiple
        activation scales and isolines. This loss measures the difference between the isoline features at different
        scales and computes a weighted mean loss.

        Parameters:
        -----------
        features_isolines : List[torch.Tensor]
                          A list of feature isoline tensors for the query, where each tensor corresponds to a layer's
                          feature isolines. Each tensor has shape `(B, N, C_i)`, where:
                          - `B`: Number of samples in the batch.
                          - `N`: Number of isolines per sample.
                          - `C_i`: Feature dimension for the respective layer.

        Returns:
        --------
        loss_batch : torch.Tensor
                   A 1D tensor of shape `(B,)` representing the total loss per sample, averaged across scales.

        loss_scales_isos_batch : torch.Tensor
                               A 2D tensor of shape `(B, N)` representing the isoline-wise loss per sample,
                               averaged across scales.
        """
        try:
            batch_size = features_isolines[0].shape[0]
            num_activations = len(self.activations)

            loss_scales = torch.zeros((batch_size, num_activations), device=self.device)
            loss_scales_isos_batch = torch.zeros(
                (batch_size, num_activations, self.nb_iso), device=self.device
            )

            if self.isoline_weights is not None:
                self.isoline_weights = self.isoline_weights.to(self.device)

            for j in range(num_activations):
                difference_features = (
                    features_isolines[j] - self.features_isolines_support[j]
                )
                c_j = difference_features.shape[-2]
                lsi = torch.norm(difference_features, dim=-2) / (c_j ** 0.5)
                loss_scales_isos_batch[:, j] = lsi
                loss_scales[:, j] = torch.mean(self.isoline_weights * lsi, dim=-1)

            loss_batch = torch.mean(loss_scales, dim=-1)
            return loss_batch, loss_scales_isos_batch

        except Exception as e:
            logger.error(f"Error computing multi-scale multi-isoline loss: {e}")
            raise

    def fit(self, img_support: torch.Tensor, polygon_support: torch.Tensor) -> None:
        """
        Fit the DCF model to the support image and contour.

        Parameters:
        -----------
        img_support : torch.Tensor
            Support image tensor of shape (B, C, H, W)
        polygon_support : torch.Tensor
            Support contour tensor of shape (B, K, 2)
        """
        if img_support.dtype != torch.float32:
            img_support = img_support.float()
        if polygon_support.dtype != torch.float32:
            polygon_support = polygon_support.float()
        try:
            size = img_support.shape[-1]
            small_size = size // (2**2)
            self.support_size = torch.tensor(
                list(img_support.shape[-2:]), dtype=torch.float32
            )
            # polygon_support arrives in [x,y] convention; torch_contour meshgrid uses [y,x]
            polygon_support_yx = torch.roll(polygon_support, shifts=1, dims=-1)

            with torch.no_grad():
                logger.info("Fitting DCF one shot...")

                self._move_model_to_device()

                img_support = img_support.to(self.device)
                polygon_support_dev = polygon_support_yx.to(self.device)

                # Compute support mask at small_size using the same Contour_to_mask
                # parameters as predict() so support and query features are directly
                # comparable (same resolution, same k).
                ctm_small = Contour_to_mask(small_size, k=1e4)
                mask_support_small = ctm_small(polygon_support_dev)
                distance_map_support_small = None

                if self.isolines is not None:
                    self.isolines = self.isolines.to(self.device)
                    self.isoline_weights = self.isoline_weights.to(self.device)
                    ctd_small = Contour_to_distance_map(small_size)
                    distance_map_support_small, _ = ctd_small(
                        polygon_support_dev, return_mask=True
                    )

                # Build feature extractor and ctf once — they hold a reference to
                # self.activations (a dict mutated in-place by hooks), so they always
                # see the latest activations without being recreated each iteration.
                if self.isolines is None:
                    self.nb_iso = 1
                    self.isoline_weights = torch.tensor(1.0, dtype=torch.float32)
                    self.ctf = Contour_to_features(small_size, self.activations)
                    class_feature_extractor = Mask_to_features(
                        self.activations
                    ).requires_grad_(False)
                else:
                    self.nb_iso = self.isolines.shape[0]
                    self.ctf = Contour_to_isoline_features(
                        small_size,
                        self.activations,
                        halfway_value=0.5,
                        isolines=self.isolines,
                    )
                    class_feature_extractor = Distance_map_to_isoline_features(
                        self.activations, halfway_value=0.5, isolines=self.isolines
                    )
                class_feature_extractor.compute_features_mask = True

                for i in tqdm(range(self.nb_augment), desc="Augmenting support"):
                    if self.isolines is None:
                        # First iteration always uses the original view so the exact
                        # support appearance is always represented in the feature average.
                        if i == 0:
                            img_iter, mask_iter = img_support, mask_support_small
                        else:
                            img_iter, mask_iter = augmentation(
                                (img_support, mask_support_small), self.augmentations
                            )
                        _ = self.model(preprocess(img_iter))
                        input_ = (mask_iter,)
                    else:
                        if i == 0:
                            img_iter = img_support
                            mask_iter = mask_support_small
                            dmap_iter = distance_map_support_small
                        else:
                            img_iter, mask_iter, dmap_iter = augmentation(
                                (img_support, mask_support_small, distance_map_support_small),
                                self.augmentations,
                            )
                        _ = self.model(preprocess(img_iter))
                        input_ = (dmap_iter, mask_iter)

                    tmp, tmp_mask = class_feature_extractor(*input_)

                    if i == 0:
                        self.features_isolines_support = tmp
                        self.features_mask_support = tmp_mask
                    else:
                        for j, (iso, mask) in enumerate(zip(tmp, tmp_mask)):
                            self.features_isolines_support[j] += iso
                            self.features_mask_support[j] += mask

                self.features_isolines_support = [
                    u / self.nb_augment for u in self.features_isolines_support
                ]
                self.features_anchor_mask = [
                    u / self.nb_augment for u in self.features_mask_support
                ]

                self.weights = torch.tensor(
                    [1 / (2) ** i for i in range(len(self.activations))],
                    dtype=torch.float32,
                    device=self.device,
                )
                self.weights = self.weights / torch.sum(self.weights)

                logger.info("DCF fitting completed successfully")

        except Exception as e:
            logger.error(f"Error during fitting: {e}")
            raise

    def similarity_score(self, features_mask_query: List[torch.Tensor]) -> torch.Tensor:
        """
        Computes a similarity score between query feature masks and support feature masks
        using a weighted cosine similarity across multiple activation layers.

        Parameters:
        -----------
        features_mask_query : List[torch.Tensor]
                            A list of query feature mask tensors, where each tensor corresponds to a layer's
                            feature representation. Each tensor has shape (B, C), where:
                            - `B`: Batch size (number of query samples).
                            - `C`: Feature dimension for the respective layer.

        Returns:
        --------
        torch.Tensor:
                    A 1D tensor of shape (B,) representing the weighted similarity score for each
                    sample in the batch across all layers.
        """
        try:
            b = features_mask_query[0].shape[0]
            num_activations = len(self.activations)
            score = torch.zeros((num_activations, b), device=self.device)

            for i in range(num_activations):
                support_features = torch.squeeze(self.features_mask_support[i])
                query_features = torch.squeeze(features_mask_query[i]).to(torch.float32)

                numerator = torch.sum(support_features * query_features, dim=-1)
                denominator = torch.linalg.norm(
                    support_features, dim=-1
                ) * torch.linalg.norm(query_features, dim=-1)
                cos = self.weights[i] * numerator / (denominator + 1e-8)
                score[i] = torch.flatten(cos)

            return torch.mean(score, dim=0)

        except Exception as e:
            logger.error(f"Error computing similarity score: {e}")
            raise

    def predict(
        self, imgs_query: torch.Tensor, contours_query: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Predicts contours for the query images using gradient descent-based optimization.
        This function refines the contours through several epochs, computes the loss and loss_scales_isos
        at each step, and returns the best contours based on the minimum loss.

        Parameters:
        -----------
        imgs_query : torch.Tensor
                   A batch of input query images of shape `(B, C, H, W)`.
        contours_query : torch.Tensor
                       Initial contour points for the images of shape `(B, 1, K, 2)`.

        where - B is the batch size,
                  - K is the number of nodes in each contour
                  - C = 3 the number of channel
                  - H the Height of the images
                  - W the Width of the images

        Returns:
        --------
        epochs_contours_query[argmin] : np.ndarray
                                      The predicted contours of shape `(B, K, 2)` that minimize the loss.
        scores : torch.Tensor
               Similarity scores for each contour, indicating how well the final query contours match
               the learned features of the support contour, with shape `(B,)`.
        losses : np.ndarray
               Losses recorded over epochs, shape `(N_epoch, B)`.
        loss_scales_isos : np.ndarray
                         Energy values recorded over epochs, representing isoline-wise losses across layers,
                         shape `(n_epochs, B, num_activations, num_isolines)`.
        """
        if imgs_query.dtype != torch.float32:
            imgs_query = imgs_query.float()
        if contours_query.dtype != torch.float32:
            contours_query = contours_query.float()
        try:
            if not hasattr(self, "ctf") or not hasattr(self, "features_isolines_support"):
                raise RuntimeError(
                    "Model must be fitted before prediction. Call fit() first."
                )

            if not hasattr(self, "nb_iso"):
                raise RuntimeError(
                    "nb_iso not defined. Model must be fitted before prediction."
                )

            b = contours_query.shape[0]
            self.nb_points = contours_query.shape[-2]
            batch_size, _, h, w = imgs_query.shape
            losses = np.zeros((self.n_epochs, batch_size))
            epochs_contours_query = np.zeros(
                (self.n_epochs, batch_size, contours_query.shape[-2], 2)
            )
            loss_scales_isos = np.zeros(
                (self.n_epochs, batch_size, len(self.activations), self.nb_iso)
            )

            scale = self.support_size.to(self.device) / torch.tensor(
                [h, w], dtype=torch.float32, device=self.device
            )

            self.img_dim = torch.tensor(
                imgs_query.shape[-2:], dtype=torch.float32, device=self.device
            )

            contours_query_array = contours_query.cpu().detach().numpy()
            contours_query_array = self.cleaner.clean_contours_and_interpolate(
                contours_query_array
            ).clip(0, 1)
            contours_query = (
                torch.from_numpy(np.roll(contours_query_array, axis=-1, shift=1))
                .float()
                .to(self.device)
            )
            contours_query.requires_grad = True

            self._move_model_to_device()
            scale = scale.to(self.device)
            imgs_query = imgs_query.to(self.device)

            _ = self.model(preprocess(imgs_query))

            logger.info("Contour is evolving please wait a few moments...")

            best_loss = float("inf")
            patience_counter = 0

            for i in tqdm(range(self.n_epochs), desc="Evolving contour"):
                if self.use_mixed_precision and self.scaler is not None:
                    with torch.amp.autocast('cuda'):
                        features_isoline_query, _ = self.ctf(contours_query)
                        loss_batch, loss_scales_isos_batch = (
                            self.multi_scale_multi_isoline_loss(features_isoline_query)
                        )
                        loss_all = b * torch.mean(
                            loss_batch
                            + self.lambda_area * area(contours_query)[:, 0]
                        )

                    self.scaler.scale(loss_all).backward(inputs=contours_query)
                    self.scaler.update()
                else:
                    features_isoline_query, _ = self.ctf(contours_query)
                    loss_batch, loss_scales_isos_batch = (
                        self.multi_scale_multi_isoline_loss(features_isoline_query)
                    )
                    loss_all = b * torch.mean(
                        loss_batch
                        + self.lambda_area * area(contours_query)[:, 0]
                    )
                    loss_all.backward(inputs=contours_query)

                current_loss = loss_all.item()
                losses[i] = loss_batch.detach().cpu().numpy()
                epochs_contours_query[i] = contours_query.detach().cpu().numpy()[:, 0]
                loss_scales_isos[i] = loss_scales_isos_batch.detach().cpu().numpy()

                if current_loss < best_loss - self.early_stopping_threshold:
                    best_loss = current_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {i + 1}")
                    break

                # Gradient must exist after backward — asserted for type checker.
                assert contours_query.grad is not None, "backward() must have run before this point"
                with torch.no_grad():
                    # Unscale so the stop threshold is consistent with/without AMP.
                    raw_grad = contours_query.grad
                    if self.scaler is not None:
                        raw_grad = raw_grad / self.scaler.get_scale()
                    norm_grad = torch.unsqueeze(torch.norm(raw_grad, dim=-1), -1)
                    stop = (torch.amax(norm_grad[:, 0], dim=-2) < self.thresh)[-1]

                if not torch.all(stop):
                    with torch.no_grad():
                        # Normalize to unit gradient per node so every active node
                        # steps by exactly lr × decay^i regardless of gradient
                        # magnitude — eliminates the "barely moves / diverges" problem
                        # caused by sparse winding-number gradients.
                        gradient_direction = raw_grad / (norm_grad + 1e-8)
                        gradient_direction = self.smooth(gradient_direction)
                        # Cap per-node displacement to self.clip so high learning
                        # rates cannot send nodes outside [0,1] and degenerate
                        # the contour into a boundary-hugging square.
                        step_size = min(self.learning_rate * (self.ed**i), self.clip)
                        contours_query = (
                            contours_query
                            - scale
                            * step_size
                            * gradient_direction
                        )
                    interpolated_contour = self.cleaner.clean_contours_and_interpolate(
                        contours_query.detach().cpu().numpy()
                    )
                    contours_query = torch.clip(
                        torch.from_numpy(interpolated_contour).float().to(self.device),
                        0,
                        1,
                    )

                    contours_query.grad = None
                    contours_query.requires_grad = True

                else:
                    logger.info("The algorithm stopped earlier")
                    break

            # Calculate score after gradient descent
            self.ctf.compute_features_mask = True
            _, features_mask_query = self.ctf(contours_query)

            scores = self.similarity_score(features_mask_query).cpu().detach().numpy()

            losses[losses == 0] = 1e10
            argmin = np.argmin(losses, axis=0)

            best_contours = epochs_contours_query[argmin, np.arange(batch_size)]

            img_dims = np.array(self.img_dim.cpu().numpy())  # [H, W]
            # Store all contour positions in [x_pixel, y_pixel] for visualization.
            # Count epochs that actually ran before zeros are replaced with 1e10.
            n_valid = int(np.sum(losses[:, 0] > 0))
            _epochs_yx_px = epochs_contours_query[:n_valid] * img_dims[None, None, None]
            self.contour_history_ = np.roll(_epochs_yx_px, axis=-1, shift=-1).astype(np.int32)

            # best_contours is in [y_norm, x_norm]; scale then swap to [x_pixel, y_pixel]
            best_contours_scaled = (
                best_contours * img_dims[None, None]
            )  # [y_pixel, x_pixel]
            best_contours_final = np.roll(
                best_contours_scaled, axis=-1, shift=-1
            ).astype(np.int32)  # [x_pixel, y_pixel]

            # Apply GrabCut if requested
            if self.do_apply_grabcut:
                logger.info("Applying GrabCut post-processing...")
                best_contours_final = self._apply_grabcut_postprocessing(
                    imgs_query, best_contours_final
                )

            logger.info("Prediction completed successfully")
            return best_contours_final, scores, losses, loss_scales_isos

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise RuntimeError(f"Prediction failed: {e}")

    def _get_model_device(self) -> torch.device:
        """Get the device of the model in a robust way."""
        try:
            if list(self.model.parameters()):
                return next(self.model.parameters()).device
            elif list(self.model.buffers()):
                return next(self.model.buffers()).device
            else:
                return torch.device("cpu")
        except Exception:
            return torch.device("cpu")

    def _move_model_to_device(self) -> None:
        """Move the model to the target device if needed."""
        try:
            current_device = self._get_model_device()
            if current_device != self.device:
                self.model = self.model.to(self.device)
        except Exception as e:
            logger.warning(f"Could not move model to device {self.device}: {e}")

    def _apply_grabcut_postprocessing(
        self, img: torch.Tensor, final_contours: np.ndarray
    ) -> np.ndarray:
        """
        Apply GrabCut post-processing to refine the final contours.

        Args:
            img: Input image tensor (B, C, H, W)
            final_contours: Final contours from DCF (B, K, 2) - already in image coordinates

        Returns:
            Refined contours after GrabCut processing
        """
        try:
            refined_contours = []

            for i in range(img.shape[0]):
                # Convert tensor to numpy
                img_np = img[i].cpu().numpy()
                img_np = np.moveaxis(img_np, 0, -1)  # (C, H, W) -> (H, W, C)
                img_np = (img_np * 255).astype(np.uint8)

                # Get contour for this batch
                contour = final_contours[i]

                # In oneshot_dcf, contours are already in image coordinates
                # Just ensure they are within bounds
                h, w = img_np.shape[:2]
                contour = np.clip(contour, 0, [w - 1, h - 1]).astype(np.int32)

                # Create mask from contour
                mask = np.zeros((h, w), dtype=np.uint8)

                # Fix contour format for cv2.fillPoly
                if len(contour.shape) == 2:
                    contour_for_fill = contour.reshape(-1, 1, 2)
                else:
                    contour_for_fill = contour

                cv2.fillPoly(mask, [contour_for_fill], 1)

                if np.sum(mask) == 0:
                    logger.warning(f"Empty mask for sample {i}, skipping GrabCut")
                    refined_contours.append(contour)
                    continue

                # Apply GrabCut
                distance_map = distance_transform_edt(mask)
                distance_map = distance_map / np.max(distance_map)
                distance_map_outside = distance_transform_edt(1 - mask)
                distance_map_outside = distance_map_outside / np.max(
                    distance_map_outside
                )

                mask_grabcut = np.full(mask.shape, cv2.GC_PR_BGD, dtype=np.uint8)
                mask_grabcut[distance_map > 0.8] = cv2.GC_FGD
                mask_grabcut[(distance_map > 0.5) & (distance_map <= 0.8)] = (
                    cv2.GC_PR_FGD
                )
                mask_grabcut[distance_map_outside > 0.8] = cv2.GC_BGD

                bgdModel = np.zeros((1, 65), np.float64)
                fgdModel = np.zeros((1, 65), np.float64)

                cv2.grabCut(
                    img_np,
                    mask_grabcut,
                    None,
                    bgdModel,
                    fgdModel,
                    5,
                    cv2.GC_INIT_WITH_MASK,
                )

                result = np.where(
                    (mask_grabcut == cv2.GC_FGD) | (mask_grabcut == cv2.GC_PR_FGD), 1, 0
                ).astype(np.uint8)

                # Get largest connected component
                labeled_array, num_features = label(result)
                if num_features > 0:
                    largest_cc = np.argmax(np.bincount(labeled_array.flat)[1:]) + 1
                    result = (labeled_array == largest_cc).astype(np.uint8)

                # Find contours from refined mask
                contours, _ = cv2.findContours(
                    result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                if contours:
                    # Get the largest contour
                    largest_contour = max(contours, key=cv2.contourArea)
                    refined_contours.append(largest_contour.reshape(-1, 2))
                else:
                    # Fallback to original contour
                    refined_contours.append(contour)

            logger.info("GrabCut post-processing completed")
            return np.array(refined_contours)

        except Exception as e:
            logger.error(f"Error in GrabCut post-processing: {e}")
            return final_contours  # Return original contours if GrabCut fails
