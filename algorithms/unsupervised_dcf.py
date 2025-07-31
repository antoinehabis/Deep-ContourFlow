import logging
import sys
from pathlib import Path
from typing import Tuple

import cv2
from scipy.ndimage import distance_transform_edt, label

from utils import Contour_to_features, piecewise_linear

sys.path.append(str(Path(__file__).resolve().parent.parent))
import numpy as np
import torch
import torchvision.models as models
from scipy import optimize
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from torch_contour import CleanContours, Smoothing, area
from torchvision import transforms
from tqdm import tqdm

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
    """
    Implementation of the unsupervised Deep Contour Flow (DCF) algorithm.

    This class implements the unsupervised version of DCF that moves the contour
    over time to push as far away as possible the features inside and outside the contour.
    """

    def __init__(
        self,
        n_epochs: int = 100,
        model: torch.nn.Module = VGG16,
        learning_rate: float = 5e-2,
        clip: float = 1e-1,
        exponential_decay: float = 0.998,
        area_force: float = 0.0,
        sigma: float = 1,
        early_stopping_patience: int = 10,
        early_stopping_threshold: float = 1e-6,
        use_mixed_precision: bool = False,
        do_apply_grabcut: bool = False,
    ):
        """
        Initialize the DCF algorithm with the specified parameters.

        Args:
            n_epochs: Maximum number of training epochs
            model: Pre-trained model for extracting activations
            learning_rate: Learning rate for optimization
            clip: Gradient clipping value
            exponential_decay: Exponential decay factor for learning rate
            area_force: Weight of the contour area constraint
            sigma: Standard deviation of the Gaussian smoothing operator
            early_stopping_patience: Number of epochs before early stopping
            early_stopping_threshold: Minimum improvement threshold for early stopping
            use_mixed_precision: Use mixed precision for GPU acceleration

        Raises:
            ValueError: If parameters are invalid
        """

        self._validate_parameters(
            n_epochs,
            learning_rate,
            clip,
            exponential_decay,
            area_force,
            sigma,
            early_stopping_patience,
            early_stopping_threshold,
        )

        self.n_epochs = n_epochs
        self.model = model
        self.learning_rate = learning_rate
        self.clip = clip
        self.ed = exponential_decay
        self.lambda_area = area_force
        self.device = None

        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.use_mixed_precision = use_mixed_precision
        self.do_apply_grabcut = do_apply_grabcut

        self._initialize_components(sigma)

        if self.use_mixed_precision:
            if not torch.cuda.is_available():
                logger.warning(
                    "Mixed precision requested but CUDA not available. Disabling."
                )
                self.use_mixed_precision = False
            else:
                self.scaler = torch.cuda.amp.GradScaler()

        logger.info(f"DCF initialized with {n_epochs} epochs, lr={learning_rate}")

    def _validate_parameters(
        self,
        n_epochs: int,
        learning_rate: float,
        clip: float,
        exponential_decay: float,
        area_force: float,
        sigma: float,
        early_stopping_patience: int,
        early_stopping_threshold: float,
    ) -> None:
        """Validate input parameters."""
        if n_epochs <= 0:
            raise ValueError("n_epochs must be positive")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if clip <= 0:
            raise ValueError("clip must be positive")
        if not 0 < exponential_decay < 1:
            raise ValueError("exponential_decay must be between 0 and 1")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative")
        if early_stopping_threshold < 0:
            raise ValueError("early_stopping_threshold must be non-negative")

    def _initialize_components(self, sigma: float) -> None:
        """Initialize algorithm components."""
        try:
            self.activations = {}
            self.shapes = {}

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
                if hasattr(self.model, str(layer_idx)):
                    self.model[layer_idx].register_forward_hook(self.get_activations(i))
                else:
                    logger.warning(f"Layer {layer_idx} not found in model")
        except Exception as e:
            logger.error(f"Error configuring hooks: {e}")
            raise

    def get_activations(self, name: int):
        """
        Create a hook to capture activations from a specific layer.

        Args:
            name: Layer name/index

        Returns:
            Hook function to capture activations
        """

        def hook(model, input, output):
            try:
                device = input[0].device
                self.activations[name] = output.to(device)
            except Exception as e:
                logger.error(f"Error capturing activations: {e}")
                raise

        return hook

    def multiscale_loss(
        self, features: Tuple[list, list], weights: torch.Tensor, eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Compute a multiscale loss based on features inside and outside the mask.

        Args:
            features: Tuple containing (features_inside, features_outside) for each scale
            weights: Weights for each scale
            eps: Small value to avoid division by zero

        Returns:
            Computed multiscale loss
        """
        try:
            batch_size = features[0][0].shape[0]
            features_inside, features_outside = features
            nb_scales = len(features_inside)
            energies = torch.zeros((nb_scales, batch_size), device=self.device)

            for j in range(nb_scales):
                diff = features_inside[j] - features_outside[j]
                norm_diff = torch.linalg.vector_norm(diff, 2, dim=-2)[..., 0]
                norm_activations = torch.linalg.vector_norm(
                    self.activations[j], 2, dim=(1, 2, 3)
                )
                norm_mse = -norm_diff / (norm_activations + eps)
                energies[j] = norm_mse

            return torch.sum(energies * weights[..., None], axis=0)

        except Exception as e:
            logger.error(f"Error computing multiscale loss: {e}")
            raise

    def predict(
        self, img: torch.Tensor, contour_init: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict contour for a given image and initial contour.

        Args:
            img: Input image tensor of shape (B, C, H, W)
            contour_init: Initial contour tensor of shape (B, 1, K, 2)

        Returns:
            Tuple containing:
            - contour_history: Contour history during prediction
            - loss_history: Loss values history
            - final_contours: Optimized final contours

        Raises:
            ValueError: If input tensors are invalid
            RuntimeError: If an error occurs during optimization
        """
        try:
            self._validate_inputs(img, contour_init)

            self.device = contour_init.device
            self.img_dim = torch.tensor(img.shape[-2:], device=self.device)

            # Prepare data
            loss_history = np.zeros((contour_init.shape[0], self.n_epochs))
            contour_history = []

            self._setup_model_and_activations(img)

            contour, optimizer, lr_scheduler = self._setup_optimization(contour_init)

            self._setup_processing_components(img)

            contour_history, loss_history = self._run_optimization_loop(
                contour, optimizer, lr_scheduler, loss_history, contour_history
            )

            final_contours = self._compute_final_contours(contour_history, loss_history)

            # Apply GrabCut if requested
            if self.do_apply_grabcut:
                logger.info("Applying GrabCut post-processing...")
                final_contours = self._apply_grabcut_postprocessing(img, final_contours)

            logger.info("Prediction completed successfully")
            return (
                np.roll(contour_history, axis=-1, shift=-1),
                loss_history,
                final_contours,
            )

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise RuntimeError(f"Prediction failed: {e}")

    def _validate_inputs(self, img: torch.Tensor, contour_init: torch.Tensor) -> None:
        """Validate input tensors."""
        if img.dtype != torch.float32:
            raise ValueError("Image must be of type float32")
        if img.dim() != 4:
            raise ValueError("Image must have 4 dimensions (B, C, H, W)")
        if contour_init.dim() != 4:
            raise ValueError("Initial contour must have 4 dimensions (B, 1, K, 2)")
        if img.shape[0] != contour_init.shape[0]:
            raise ValueError("Image and contour batch sizes must match")

    def _setup_model_and_activations(self, img: torch.Tensor) -> None:
        """Configure model and extract activations."""
        try:
            # Move model to appropriate device
            if str(self.device) == "cuda:0":
                self.model = self.model.cuda()
            elif str(self.device) == "mps:0":
                self.model = self.model.to(torch.device("mps"))

            # Extract activations
            with torch.no_grad():
                _ = self.model(preprocess(img))

        except Exception as e:
            logger.error(f"Error configuring model: {e}")
            raise

    def _setup_optimization(
        self, contour_init: torch.Tensor
    ) -> Tuple[torch.Tensor, Adam, ExponentialLR]:
        """Configure optimization."""
        try:
            contour = torch.roll(contour_init, dims=-1, shifts=1)
            contour.requires_grad = True

            optimizer = Adam([contour], lr=self.learning_rate, eps=1e-8)
            lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=self.ed)

            return contour, optimizer, lr_scheduler

        except Exception as e:
            logger.error(f"Error configuring optimization: {e}")
            raise

    def _setup_processing_components(self, img: torch.Tensor) -> None:
        """Configure processing components."""
        try:
            self.ctf = Contour_to_features(img.shape[-1] // (2**2), self.activations)

            self.weights = torch.tensor(
                [1 / (2**i) for i in range(len(self.activations))],
                device=self.device,
                dtype=torch.float32,
            )
            self.weights = self.weights / torch.sum(self.weights)
            self.weights.requires_grad = False

        except Exception as e:
            logger.error(f"Error configuring processing components: {e}")
            raise

    def _run_optimization_loop(
        self,
        contour: torch.Tensor,
        optimizer: Adam,
        lr_scheduler: ExponentialLR,
        loss_history: np.ndarray,
        contour_history: list,
    ) -> Tuple[list, np.ndarray]:
        """Execute main optimization loop."""
        try:
            best_loss = float("inf")
            patience_counter = 0

            logger.info("Starting contour evolution...")

            for i in tqdm(range(self.n_epochs), desc="Optimizing contour"):
                optimizer.zero_grad()

                loss, batch_loss = self._compute_loss(contour)

                self._backward_and_update(loss, contour, optimizer)
                lr_scheduler.step()

                contour = self._smooth_contour(contour)

                contour_cleaned = self._save_history(
                    contour, batch_loss, loss_history, contour_history, i
                )

                optimizer.param_groups[0]["params"][0] = contour_cleaned
                contour = contour_cleaned

                if self._check_early_stopping(batch_loss, best_loss, patience_counter):
                    logger.info(f"Early stopping at epoch {i+1}")
                    break

                best_loss, patience_counter = self._update_early_stopping_vars(
                    batch_loss, best_loss, patience_counter
                )

            return contour_history, loss_history

        except Exception as e:
            logger.error(f"Error during optimization loop: {e}")
            raise

    def _compute_loss(self, contour: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute loss for current step."""
        try:
            if self.use_mixed_precision and str(self.device) == "cuda:0":
                with torch.cuda.amp.autocast():
                    features = self.ctf(contour)
                    batch_loss = (
                        self.multiscale_loss(features, self.weights)
                        + self.lambda_area * area(contour)[:, 0]
                    )
                    loss = self.img_dim[0] * torch.mean(batch_loss)
            else:
                features = self.ctf(contour)
                batch_loss = (
                    self.multiscale_loss(features, self.weights)
                    + self.lambda_area * area(contour)[:, 0]
                )
                loss = self.img_dim[0] * torch.mean(batch_loss)

            return loss, batch_loss

        except Exception as e:
            logger.error(f"Error computing loss: {e}")
            raise

    def _backward_and_update(
        self, loss: torch.Tensor, contour: torch.Tensor, optimizer: Adam
    ) -> None:
        """Perform backward pass and parameter update."""
        try:
            if self.use_mixed_precision and str(self.device) == "cuda:0":
                self.scaler.scale(loss).backward(inputs=contour)
                self.scaler.unscale_(optimizer)
                clip_grad_norm_(contour, self.clip)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward(inputs=contour)
                clip_grad_norm_(contour, self.clip)
                optimizer.step()

        except Exception as e:
            logger.error(f"Error during backward pass: {e}")
            raise

    def _smooth_contour(self, contour: torch.Tensor) -> torch.Tensor:
        """Apply smoothing to contour."""
        try:
            contour_input = torch.clone(contour)
            return (
                self.smooth((contour - contour_input).to(torch.float32)) + contour_input
            )
        except Exception as e:
            logger.error(f"Error smoothing contour: {e}")
            raise

    def _save_history(
        self,
        contour: torch.Tensor,
        batch_loss: torch.Tensor,
        loss_history: np.ndarray,
        contour_history: list,
        epoch: int,
    ) -> torch.Tensor:
        """Save optimization history and return cleaned contour."""
        try:
            with torch.no_grad():
                loss_history[:, epoch] = batch_loss.cpu().detach().numpy()

                contour_np = contour.cpu().detach().numpy()
                # Scale to original image dimensions - correct order: [width, height] for [x, y] coordinates
                img_dims = np.array(self.img_dim.cpu().numpy())  # [height, width]
                # Swap to [width, height] for [x, y] coordinates
                img_dims_xy = img_dims[::-1]  # [width, height]
                contour_scaled = contour_np * img_dims_xy[None, None, None]
                contour_history.append(contour_scaled.astype(np.int32))

                contour_without_loops = self.cleaner.clean_contours_and_interpolate(
                    contour_np
                )
                contour_cleaned = torch.clip(
                    torch.from_numpy(contour_without_loops), 0, 1
                )

                if str(self.device) == "cuda:0":
                    contour_cleaned = contour_cleaned.to(torch.float32).cuda()
                elif str(self.device) == "mps:0":
                    contour_cleaned = contour_cleaned.to(torch.float32).to(
                        torch.device("mps")
                    )

                contour_cleaned.grad = None
                contour_cleaned.requires_grad = True

                return contour_cleaned

        except Exception as e:
            logger.error(f"Error saving history: {e}")
            raise

    def _check_early_stopping(
        self, batch_loss: torch.Tensor, best_loss: float, patience_counter: int
    ) -> bool:
        """Check if early stopping should be triggered."""
        return patience_counter >= self.early_stopping_patience

    def _update_early_stopping_vars(
        self, batch_loss: torch.Tensor, best_loss: float, patience_counter: int
    ) -> Tuple[float, int]:
        """Update early stopping variables."""
        current_loss = torch.mean(batch_loss).item()
        if current_loss < best_loss - self.early_stopping_threshold:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1
        return best_loss, patience_counter

    def _compute_final_contours(
        self, contour_history: list, loss_history: np.ndarray
    ) -> np.ndarray:
        """Compute optimized final contours."""
        try:
            contour_history_array = np.roll(
                np.stack(contour_history), axis=-1, shift=-1
            )[:, :, 0]

            final_contours = np.zeros(
                (
                    loss_history.shape[0],
                    contour_history_array.shape[-2],
                    contour_history_array.shape[-1],
                )
            )

            for i, loss in enumerate(loss_history):
                try:
                    # Remove NaN values from loss history
                    valid_loss = loss[~np.isnan(loss)]
                    if len(valid_loss) < 2:
                        logger.warning(
                            f"Not enough valid loss values for sample {i}, using last contour"
                        )
                        final_contours[i] = contour_history_array[-1, i]
                        continue

                    p, _ = optimize.curve_fit(
                        piecewise_linear,
                        np.arange(len(valid_loss)),
                        valid_loss,
                        bounds=(
                            np.array([0, -np.inf, -np.inf, -np.inf]),
                            np.array([len(valid_loss), np.inf, np.inf, np.inf]),
                        ),
                    )

                    index_stop = int(p[0]) - 10
                    index_stop = max(0, min(index_stop, len(contour_history_array) - 1))

                    final_contours[i] = contour_history_array[index_stop, i]

                except Exception as e:
                    logger.warning(f"Error computing final contour for sample {i}: {e}")
                    final_contours[i] = contour_history_array[-1, i]

            logger.info("Contour stopped")
            return final_contours

        except Exception as e:
            logger.error(f"Error computing final contours: {e}")
            raise

    def _apply_grabcut_postprocessing(
        self, img: torch.Tensor, final_contours: np.ndarray
    ) -> np.ndarray:
        """
        Apply GrabCut post-processing to refine the final contours.

        Args:
            img: Input image tensor (B, C, H, W)
            final_contours: Final contours from DCF (B, K, 2)

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

                # Create mask from contour
                mask = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=np.uint8)

                # Fix contour format for cv2.fillPoly
                if len(contour.shape) == 2:
                    contour_for_fill = contour.reshape(-1, 1, 2).astype(int)
                else:
                    contour_for_fill = contour.astype(int)

                cv2.fillPoly(mask, [contour_for_fill], 1)

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
