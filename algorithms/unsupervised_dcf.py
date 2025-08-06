import logging
import sys
from pathlib import Path
from typing import Tuple

from algorithms.utils import apply_grabcut_postprocessing_parallel
from models.models import (
    VGG16,
    create_model,
    detect_model_type,
    get_model_layer_access,
    get_model_layer_indices,
    get_model_preprocess,
)
from utils import Contour_to_features, piecewise_linear

sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from scipy import optimize
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_contour import CleanContours, Smoothing, area
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DCF:
    """
    Implementation of the unsupervised Deep Contour Flow (DCF) algorithm.

    This class implements the unsupervised version of DCF that moves the contour
    over time to push as far away as possible the features inside and outside the contour.
    """

    def __init__(
        self,
        n_epochs: int = 50,  # Réduit de 100 à 50
        model=VGG16,  # Peut être un torch.nn.Module ou un string
        learning_rate: float = 1e-1,  # Augmenté pour convergence plus rapide
        clip: float = 5e-2,  # Réduit pour stabilité
        exponential_decay: float = 0.998,
        area_force: float = 0.0,
        sigma: float = 1,
        early_stopping_patience: int = 5,  # Réduit pour arrêt plus rapide
        early_stopping_threshold: float = 1e-6,
        use_mixed_precision: bool = True,  # Activé par défaut
        do_apply_grabcut: bool = False,
        max_batch_size: int = 4,  # Nouveau paramètre pour le traitement par batch
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
            do_apply_grabcut: Apply GrabCut post-processing
            max_batch_size: Maximum batch size for processing

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
        # Initialiser le modèle
        self.model = self._initialize_model(model)
        self.model_type = detect_model_type(self.model)
        self.learning_rate = learning_rate
        self.clip = clip
        self.ed = exponential_decay
        self.lambda_area = area_force
        self.device = None
        self.max_batch_size = max_batch_size

        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.use_mixed_precision = use_mixed_precision
        self.do_apply_grabcut = do_apply_grabcut

        # 1. Optimisations GPU
        self._setup_gpu_optimizations()

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

    def _initialize_model(self, model) -> torch.nn.Module:
        """
        Initialise le modèle en fonction du paramètre fourni.

        Args:
            model: PyTorch model, model class, or string specifying the model type

        Returns:
            Modèle initialisé
        """
        if isinstance(model, str):
            return create_model(model)
        elif isinstance(model, type) and issubclass(model, torch.nn.Module):
            # If it's a model class, create an instance
            return model()
        else:
            # If it's already a model instance
            return model

    def _setup_gpu_optimizations(self):
        """Configure GPU optimizations for better performance."""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            logger.info("GPU optimizations enabled")

    def _cleanup_gpu_memory(self):
        """Clean up GPU memory cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
            # For models that return features directly, no hooks needed
            if self.model_type in [
                "resnet_fpn",
                "resnet50",
                "resnet101",
                "resnet101_fpn",
            ]:
                logger.info(
                    f"{self.model_type} detected: no hooks needed, activations will be captured in forward pass"
                )
                return

            # Determine layer indices to use based on model type
            layer_indices = get_model_layer_indices(self.model_type)
            layer_access = get_model_layer_access(self.model_type)

            for i, layer_idx in enumerate(layer_indices):
                layer_model = layer_access(self.model, layer_idx)

                if layer_model is not None:
                    layer_model.register_forward_hook(self.get_activations(i))
                else:
                    logger.warning(
                        f"Layer {layer_idx} not found in model or not accessible."
                    )
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
        Optimized version with vectorized operations where possible.

        Args:
            features: Tuple containing (features_inside, features_outside) for each scale
            weights: Weights for each scale
            eps: Small value to avoid division by zero

        Returns:
            Computed multiscale loss
        """
        try:
            features_inside, features_outside = features
            nb_scales = len(features_inside)
            batch_size = features_inside[0].shape[0]
            energies = torch.zeros((nb_scales, batch_size), device=self.device)

            # 2. Optimisations de calcul - Traitement par scale avec gestion des tailles différentes
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
                img_np = img.cpu().numpy()
                final_contours = apply_grabcut_postprocessing_parallel(
                    img_np, final_contours
                )

            logger.info("Prediction completed successfully")
            return (
                np.roll(contour_history, axis=-1, shift=-1),
                loss_history,
                final_contours,
            )

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise RuntimeError(f"Prediction failed: {e}")

    def _process_single_batch(
        self, imgs: torch.Tensor, contours_init: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process a single batch of images and contours."""
        return self.predict(imgs, contours_init)

    def _merge_batch_results(
        self, results: list
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Merge results from multiple batches."""
        contour_histories = []
        loss_histories = []
        final_contours = []

        for contour_history, loss_history, final_contour in results:
            contour_histories.append(contour_history)
            loss_histories.append(loss_history)
            final_contours.append(final_contour)

        return (
            np.concatenate(contour_histories, axis=0),
            np.concatenate(loss_histories, axis=0),
            np.concatenate(final_contours, axis=0),
        )

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

            # Get preprocessing function for the model type
            preprocess_fn = get_model_preprocess(self.model_type)

            # Extract activations based on model type
            with torch.no_grad():
                if self.model_type in [
                    "resnet_fpn",
                    "resnet50",
                    "resnet101",
                    "resnet101_fpn",
                ]:
                    # For these models, forward returns multi-scale features directly
                    activations = self.model(preprocess_fn(img))
                    for i, activation in enumerate(activations):
                        self.activations[i] = activation.to(self.device)
                else:
                    # For other models (VGG), use normal forward pass
                    _ = self.model(preprocess_fn(img))

        except Exception as e:
            logger.error(f"Error configuring model: {e}")
            raise

    def _setup_optimization(
        self, contour_init: torch.Tensor
    ) -> Tuple[torch.Tensor, Adam, ReduceLROnPlateau]:
        """Configure optimization with improved learning rate scheduling."""
        try:
            contour = torch.roll(contour_init, dims=-1, shifts=1)
            contour = contour.contiguous()
            contour.requires_grad = True

            optimizer = Adam(
                [contour], lr=self.learning_rate, eps=1e-8, betas=(0.9, 0.999)
            )
            lr_scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6
            )

            return contour, optimizer, lr_scheduler

        except Exception as e:
            logger.error(f"Error configuring optimization: {e}")
            raise

    def _setup_processing_components(self, img: torch.Tensor) -> None:
        """Configure processing components."""
        try:
            self.ctf = Contour_to_features(img.shape[-1] // (2**2), self.activations)
            if self.model_type != "vgg16":
                self.weights = torch.tensor(
                    [1.0, 1 / 4.0, 1 / 8, 1 / 16],
                    device=self.device,
                    dtype=torch.float32,
                )
            else:
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
        lr_scheduler: ReduceLROnPlateau,
        loss_history: np.ndarray,
        contour_history: list,
    ) -> Tuple[list, np.ndarray]:
        """Execute main optimization loop with performance monitoring."""
        try:
            best_loss = float("inf")
            patience_counter = 0

            logger.info("Starting contour evolution...")

            for i in tqdm(range(self.n_epochs), desc="Optimizing contour"):
                optimizer.zero_grad()

                loss, batch_loss = self._compute_loss(contour)

                self._backward_and_update(loss, contour, optimizer)
                lr_scheduler.step(loss)  # Use loss for scheduler step

                contour = self._smooth_contour(contour)

                contour_cleaned = self._save_history(
                    contour, batch_loss, loss_history, contour_history, i
                )

                optimizer.param_groups[0]["params"][0] = contour_cleaned
                contour = contour_cleaned

                if self._check_early_stopping(batch_loss, best_loss, patience_counter):
                    logger.info(f"Early stopping at epoch {i + 1}")
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
        """Save optimization history and return cleaned contour with optimized GPU memory management."""
        try:
            with torch.no_grad():
                # Force contiguous copy to avoid negative strides
                batch_loss_contiguous = batch_loss.contiguous()
                loss_history[:, epoch] = batch_loss_contiguous.cpu().detach().numpy()

                img_dims = self.img_dim.cpu().numpy()
                img_dims_xy = img_dims[::-1].copy()
                contour_scaled = (
                    contour
                    * torch.tensor(
                        img_dims_xy, device=self.device, dtype=torch.float32
                    )[None, None, None]
                )

                contour_scaled_contiguous = contour_scaled.contiguous()
                contour_history.append(
                    contour_scaled_contiguous.cpu().detach().numpy().astype(np.int32)
                )

                # Clean up GPU memory
                contour_np = contour.cpu().detach().numpy()
                contour_cleaned_np = self.cleaner.clean_contours_and_interpolate(
                    contour_np
                )
                contour_cleaned = torch.clip(torch.from_numpy(contour_cleaned_np), 0, 1)

                if str(self.device) == "cuda:0":
                    contour_cleaned = contour_cleaned.to(torch.float32).cuda()
                elif str(self.device) == "mps:0":
                    contour_cleaned = contour_cleaned.to(torch.float32).to(
                        torch.device("mps")
                    )

                contour_cleaned.grad = None
                contour_cleaned.requires_grad = True

                # Clean up GPU memory
                self._cleanup_gpu_memory()

                return contour_cleaned

        except Exception as e:
            logger.error(f"Error saving history: {e}")
            raise

    def _check_early_stopping(
        self, batch_loss: torch.Tensor, best_loss: float, patience_counter: int
    ) -> bool:
        """Check if early stopping should be triggered with improved criteria."""
        current_loss = torch.mean(batch_loss).item()

        loss_improvement = current_loss < best_loss - self.early_stopping_threshold
        if loss_improvement:
            best_loss = current_loss
            patience_counter = 0
        else:
            patience_counter += 1

        return patience_counter >= self.early_stopping_patience

    def _update_early_stopping_vars(
        self, batch_loss: torch.Tensor, best_loss: float, patience_counter: int
    ) -> Tuple[float, int]:
        """Update early stopping variables with improved logic."""
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
