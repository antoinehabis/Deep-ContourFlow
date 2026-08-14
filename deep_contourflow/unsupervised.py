import contextlib
import logging
import warnings
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    ReduceLROnPlateau,
)
from torch_contour import (
    CleanContours,
    Smoothing,
    area,
    normals,
    sample_features_on_contour,
)
from tqdm import tqdm

from .features import Contour_to_features
from .knee import knee_index
from .models.models import (
    VGG16,
    create_model,
    detect_model_type,
    get_model_layer_access,
    get_model_layer_indices,
    get_model_preprocess,
)
from .postprocessing import apply_grabcut_fixed_length

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", message=".*padding='same'.*even kernel.*", category=UserWarning)


class DCF:
    """
    Implementation of the unsupervised Deep Contour Flow (DCF) algorithm.

    This class implements the unsupervised version of DCF that moves the contour
    over time to push as far away as possible the features inside and outside the contour.
    """

    def __init__(
        self,
        n_epochs: int = 250,
        model=VGG16,  # torch.nn.Module instance, class, or string (e.g. "vgg16")
        learning_rate: float = 1e-2,
        clip: float = 2e-1,
        area_force: float = -2e-2,  # negative = balloon expansion (outward pressure)
        sigma: float = 0.9,
        early_stopping_patience: int = 20,
        early_stopping_threshold: float = 1e-6,
        use_mixed_precision: bool = True,
        do_apply_grabcut: bool = True,
        compile: bool = True,
        process_size: int = 384,
        exponential_decay: Optional[float] = None,
        scale_weights: Union[str, list, tuple] = "uniform",
        lr_schedule: Optional[str] = "plateau",
        lr_restart_period: int = 50,
        edge_balloon: float = 0.0,
        edge_gate: float = 3.0,
        edge_attract: float = 0.0,
    ):
        """
        Initialize the DCF algorithm with the specified parameters.

        The defaults are the configuration validated on the foreground-extraction
        benchmark (ECSSD + MSRA-B + CUB-200-2011); see experiments/ and the README.
        They were tuned on 384x384 natural images, so they are a sensible starting
        point rather than a universal optimum -- other domains (histology,
        dermoscopy) may want their own sweep.

        Args:
            n_epochs: Maximum number of contour-evolution steps
            model: Pre-trained model for extracting activations
            learning_rate: Learning rate for optimization
            clip: Maximum contour displacement per step (normalized)
            area_force: Weight of the area term, *relative* to the per-sample
                separation energy. Negative expands the contour (balloon
                pressure), positive shrinks it. Dimensionless, useful range is
                roughly [-0.05, 0]; large magnitudes degrade badly.
            sigma: Standard deviation of the Gaussian smoothing operator
            early_stopping_patience: Flat-loss steps before a sample is frozen
            early_stopping_threshold: Minimum improvement threshold for early stopping
            use_mixed_precision: Use mixed precision for GPU acceleration
            do_apply_grabcut: Apply GrabCut post-processing
            process_size: Internal resolution, so convergence does not depend on
                the input size
            scale_weights: Per-scale weighting of the multiscale energy
            lr_schedule: "plateau" | "cosine" | "exponential"
            edge_balloon, edge_gate, edge_attract: experimental edge terms,
                disabled by default (neutral on the benchmark)

        Raises:
            ValueError: If parameters are invalid
        """

        self._validate_parameters(
            n_epochs,
            learning_rate,
            clip,
            area_force,
            sigma,
            early_stopping_patience,
            early_stopping_threshold,
        )

        self.n_epochs = n_epochs
        self.model = self._initialize_model(model)
        self.model_type = detect_model_type(self.model)
        self.learning_rate = learning_rate
        self.clip = clip
        self.lambda_area = area_force
        self.device = None

        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        self.use_mixed_precision = use_mixed_precision
        self.do_apply_grabcut = do_apply_grabcut
        self.compile = compile  # torch.compile the Contour_to_mask forward (tc >=1.4.5)
        # Internal processing resolution: the contour lives in [0,1] (size-agnostic),
        # but the VGG features and the mask (size//4) scale with the input, making the
        # convergence dynamics image-size-dependent. Run everything at a fixed size so
        # the behaviour is size-independent (only the resize interpolation differs).
        # None keeps the native resolution. 384 == the eval size (no change there).
        self.process_size = process_size
        # Learning-rate schedule.
        #   lr_schedule = "plateau"     -> ReduceLROnPlateau (eval-validated default);
        #                                  only decays when the global loss plateaus,
        #                                  so nodes still making progress keep full LR.
        #                 "exponential" -> ExponentialLR (lr *= exponential_decay each
        #                                  epoch). GLOBAL fixed decay: slows every node,
        #                                  incl. those still far from the object.
        #                 "cosine"      -> CosineAnnealingWarmRestarts: LR cycles down
        #                                  then RESTARTS high every lr_restart_period
        #                                  epochs, re-mobilizing nodes not yet converged.
        #   lr_schedule = None (default): back-compat -> "exponential" if
        #                 exponential_decay is set, else "plateau".
        # Note: Adam already adapts per-parameter (converged nodes -> grad~0 -> tiny
        # step; far nodes -> persistent grad -> ~lr step), so a global decay can fight
        # that. "plateau"/"cosine" preserve it better than "exponential".
        self.exponential_decay = exponential_decay
        self.lr_schedule = lr_schedule
        self.lr_restart_period = lr_restart_period
        # Per-scale weighting of the multiscale separation energy. VGG scales are
        # ordered shallow->deep (index 0 = conv1_2 texture ... 4 = conv5_3 semantics).
        # The energy prefers the most-homogeneous region, and shallow texture layers
        # make a coherent sub-PART (e.g. a shirt) win over the whole object; weighting
        # DEEP semantic layers more moves the energy minimum toward the whole object.
        #   "uniform" (default) | "linear" (∝ depth rank) | "quad" (∝ rank²) |
        #   "deepK" (only the K deepest, e.g. "deep2"/"deep3") | a custom list of weights.
        self.scale_weights = scale_weights
        # Edge-aware balloon (Option B for the part-vs-whole collapse). An outward
        # per-node force, along the contour normal, GATED by an edge-stopping function
        # so it vanishes where the contour sits on a strong feature edge (a node
        # already on the object boundary = "converged") and pushes in flat regions
        # (nodes not yet at the boundary). edge_balloon = strength (0 = off);
        # edge_gate = sharpness of the gate g = exp(-edge_gate * edge_norm). This does
        # NOT need to detect settled nodes (immune to resampling churn) — it simply
        # doesn't push nodes that are already on an edge.
        self.edge_balloon = edge_balloon
        self.edge_gate = edge_gate
        # Bidirectional edge ATTRACTION (geodesic term): pulls each node toward the
        # nearest edge ridge along its normal and is ZERO at the ridge, so nodes STOP
        # at the object boundary (handles both under- and over-shoot), unlike the
        # one-way balloon. 0 = off.
        self.edge_attract = edge_attract
        self._edge_map = None
        self._edge_grad_map = None
        self._setup_gpu_optimizations()

        self._initialize_components(sigma)

        if self.use_mixed_precision:
            if not torch.cuda.is_available():
                logger.warning(
                    "Mixed precision requested but CUDA not available. Disabling."
                )
                self.use_mixed_precision = False
            else:
                self.scaler = torch.amp.GradScaler('cuda')

        logger.info(f"DCF initialized with {n_epochs} epochs, lr={learning_rate}")

    def _initialize_model(self, model) -> torch.nn.Module:
        """Return a model instance from a module instance, class, or string name."""
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

    def _scale_weights(self, nb_scales: int) -> torch.Tensor:
        """Per-scale weights for multiscale_loss (see self.scale_weights). (nb_scales, 1).

        Scales are ordered shallow->deep; larger weight on higher indices favours
        deep/semantic layers, moving the energy minimum toward the whole object.
        """
        sw = self.scale_weights
        if isinstance(sw, (list, tuple, np.ndarray)):
            w = np.asarray(sw, dtype=np.float64)
            if len(w) != nb_scales:
                raise ValueError(
                    f"scale_weights list must have {nb_scales} entries, got {len(w)}"
                )
        elif sw == "linear":
            w = np.arange(1, nb_scales + 1, dtype=np.float64)
        elif sw == "quad":
            w = np.arange(1, nb_scales + 1, dtype=np.float64) ** 2
        elif isinstance(sw, str) and sw.startswith("deep") and sw[4:].isdigit():
            n = min(int(sw[4:]), nb_scales)
            w = np.zeros(nb_scales); w[-n:] = 1.0
        else:  # "uniform" (default) or unknown
            w = np.ones(nb_scales, dtype=np.float64)
        w = w / w.sum()
        return torch.tensor(w, dtype=torch.float32, device=self.device).reshape(nb_scales, 1)

    def multiscale_loss(
        self, features: Tuple[list, list], eps: float = 1e-6
    ) -> torch.Tensor:
        """
        Compute a multiscale loss based on features inside and outside the mask.
        Optimized version with vectorized operations where possible.

        Args:
            features: Tuple containing (features_inside, features_outside) for each scale
            eps: Small value to avoid division by zero

        Returns:
            Computed multiscale loss
        """
        try:
            features_inside, features_outside = features
            nb_scales = len(features_inside)
            batch_size = features_inside[0].shape[0]
            energies = torch.zeros((nb_scales, batch_size), device=self.device)

            for j in range(nb_scales):
                diff = features_inside[j] - features_outside[j]  # (B, C, 1) pooled means
                # Standardize the mean-difference per channel by the whole-image
                # feature std (sqrt of the spatial variance). The separation becomes
                # an effect-size (difference in units of spread), directly comparable
                # across images regardless of the feature range.
                std_c = self.activations[j].std(dim=(2, 3), unbiased=False)  # (B, C)
                d = diff[..., 0] / (std_c + eps)  # (B, C)
                energies[j] = -torch.linalg.vector_norm(d, 2, dim=-1)  # (B,)

            # ---- Scale weighting (uniform default; deep-favoring shifts the energy
            # minimum toward the whole object; see self.scale_weights) ----
            return torch.sum(energies * self._scale_weights(nb_scales), dim=0)
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
            # Output is scaled to the NATIVE image size (contours map back to the
            # user's pixels); keep img_dim = original size for that.
            self.img_dim = torch.tensor(img.shape[-2:], device=self.device)
            # Cache the (H, W) pixel-scale tensor once (used every epoch in
            # _save_history) instead of rebuilding it via a CPU roundtrip per step.
            self._hist_scale = self.img_dim.to(torch.float32)[None, None, None]

            # Process at a fixed canonical resolution so the evolution dynamics are
            # image-size-independent (VGG features + mask size no longer scale with
            # the input). The contour stays in [0,1] the whole time. GrabCut still
            # runs on the native-resolution image below.
            img_proc = img
            if self.process_size is not None and (
                img.shape[-2] != self.process_size or img.shape[-1] != self.process_size
            ):
                img_proc = F.interpolate(
                    img, size=(self.process_size, self.process_size),
                    mode="bilinear", align_corners=False,
                )

            # Prepare data
            loss_history = np.zeros((contour_init.shape[0], self.n_epochs))
            contour_history = []

            self._setup_model_and_activations(img_proc)

            contour, optimizer, lr_scheduler = self._setup_optimization(contour_init)

            self._setup_processing_components(img_proc)

            contour_history, loss_history = self._run_optimization_loop(
                contour, optimizer, lr_scheduler, loss_history, contour_history
            )

            final_contours = self._compute_final_contours(contour_history, loss_history)

            # Apply GrabCut if requested
            if self.do_apply_grabcut:
                logger.info("Applying GrabCut post-processing...")
                img_np = img.cpu().numpy()
                final_contours = apply_grabcut_fixed_length(img_np, final_contours)

            # Single allocator cleanup per image (was per-epoch in _save_history).
            self._cleanup_gpu_memory()

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
            self.model = self.model.to(self.device)

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
    ) -> Tuple[torch.Tensor, Adam, Union[ReduceLROnPlateau, ExponentialLR, CosineAnnealingWarmRestarts]]:
        """Configure optimization with improved learning rate scheduling."""
        try:
            contour = torch.roll(contour_init, dims=-1, shifts=1)
            contour = contour.contiguous()
            contour.requires_grad = True

            optimizer = Adam(
                [contour], lr=self.learning_rate, eps=1e-8, betas=(0.9, 0.999)
            )
            schedule = self.lr_schedule
            if schedule is None:  # back-compat
                schedule = "exponential" if self.exponential_decay is not None else "plateau"

            if schedule == "cosine":
                # LR decays over a cycle then restarts high, re-mobilizing nodes that
                # have not yet converged (T_mult=2 -> each cycle twice as long).
                lr_scheduler = CosineAnnealingWarmRestarts(
                    optimizer, T_0=max(1, self.lr_restart_period), T_mult=2, eta_min=1e-6
                )
            elif schedule == "exponential":
                gamma = self.exponential_decay if self.exponential_decay is not None else 0.99
                lr_scheduler = ExponentialLR(optimizer, gamma=gamma)
            else:  # "plateau"
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
            # torch-contour >=1.4.0 registers the pixel mesh as a buffer, so the
            # module must be moved to the contour's device (the mesh no longer
            # follows the input device automatically at forward time).
            self.ctf = Contour_to_features(
                img.shape[-1] // 4, self.activations, compile=self.compile
            ).to(self.device)
            # Smoothing.kernel is a registered buffer in torch-contour >=1.4.0.
            self.smooth = self.smooth.to(self.device)
            # Edge map for the edge-aware forces (fixed: activations are constant).
            if self.edge_balloon > 0 or self.edge_attract > 0:
                self._edge_map = self._compute_edge_map()
        except Exception as e:
            logger.error(f"Error configuring processing components: {e}")
            raise

    def _compute_edge_map(self, eps: float = 1e-6) -> torch.Tensor:
        """Per-image feature-edge magnitude map (B,1,H,W), normalized to ~[0,1].

        Uses the DEEPEST activation (semantic boundaries = the object outline) so the
        edge-aware balloon stops at the whole-object boundary, not internal texture
        edges. Interpolated to the mask resolution used by Contour_to_mask.
        """
        with torch.no_grad():
            act = self.activations[len(self.activations) - 1].float()  # (B,C,h,w) deepest
            a = act.mean(dim=1, keepdim=True)  # (B,1,h,w) channel-mean
            gy = a[:, :, 1:, :] - a[:, :, :-1, :]
            gx = a[:, :, :, 1:] - a[:, :, :, :-1]
            gy = torch.nn.functional.pad(gy, (0, 0, 0, 1))
            gx = torch.nn.functional.pad(gx, (0, 1, 0, 0))
            edge = torch.sqrt(gx ** 2 + gy ** 2)  # (B,1,h,w)
            edge = edge / (edge.amax(dim=(2, 3), keepdim=True) + eps)  # per-image [0,1]
            size = self.ctf.ctm.size
            edge = torch.nn.functional.interpolate(
                edge, size=(size, size), mode="bilinear", align_corners=False
            )
            # Gradient of a BLURRED edge indicator, for the edge-ATTRACTION force
            # (wider basin so nodes are pulled toward the ridge from further away).
            # Channels: [d/dx (width), d/dy (height)] to match the (x, y) grid_sample
            # convention used by sample_features_on_contour.
            eb = torch.nn.functional.avg_pool2d(edge, kernel_size=5, stride=1, padding=2)
            egx = torch.nn.functional.pad(eb[:, :, :, 1:] - eb[:, :, :, :-1], (0, 1, 0, 0))
            egy = torch.nn.functional.pad(eb[:, :, 1:, :] - eb[:, :, :-1, :], (0, 0, 0, 1))
            self._edge_grad_map = torch.cat([egx, egy], dim=1)  # (B, 2, size, size)
        return edge

    def _edge_balloon_grad(self, contour: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Outward per-node balloon force, gated by the edge map, as a gradient add.

        Returns a tensor shaped like contour.grad; adding it makes the optimizer step
        each node OUTWARD by ~edge_balloon * g, where g = exp(-edge_gate * edge_norm)
        vanishes on strong edges (nodes already on the object boundary).
        """
        with torch.no_grad():
            n = normals(contour)  # (B,1,K,2) unit outward normal (flip=0, matches ctm)
            # The DCF contour is stored (y, x) (matches Contour_to_mask), but
            # sample_features_on_contour -> grid_sample wants (x, y); flip so it
            # samples the correct pixel instead of the diagonal-transposed one.
            edge_node = sample_features_on_contour(self._edge_map, contour.flip(-1))[..., 0]  # (B,1,K)
            g = torch.exp(-self.edge_gate * edge_node).unsqueeze(-1)  # (B,1,K,1) ~1 flat, ~0 edge
            # update = -lr*grad, so -normal in grad -> +normal (outward) step.
            return -self.edge_balloon * g * n

    def _edge_attract_grad(self, contour: torch.Tensor) -> torch.Tensor:
        """Bidirectional edge-ATTRACTION force as a gradient add.

        Pulls each node toward the nearest edge ridge along its normal: force ∝
        (∇edge · n) n. It is zero AT the ridge (∇edge=0) and points inward or outward
        depending on which side the edge is — so nodes STOP at the object boundary
        (unlike the one-way balloon). Geodesic active-contour edge term.
        """
        with torch.no_grad():
            n = normals(contour)  # (B,1,K,2) in the contour's (y, x) slot order
            # Contour is (y, x); sample_features_on_contour -> grid_sample wants
            # (x, y), so flip before sampling (else the wrong pixel is read).
            ge = sample_features_on_contour(self._edge_grad_map, contour.flip(-1))  # (B,1,K,2)=[d/dx,d/dy]
            # ge channels are [d/dx, d/dy]; reorder to [d/dy, d/dx] so the dot
            # product below pairs with n's (y, x) components instead of crossing them.
            ge = ge.flip(-1)  # -> [d/dy, d/dx]
            proj = (ge * n).sum(dim=-1, keepdim=True)  # (B,1,K,1) edge-grad along normal
            # update = -lr*grad, so grad=-proj*n -> step +proj*n (toward higher edge).
            return -self.edge_attract * proj * n

    def _run_optimization_loop(
        self,
        contour: torch.Tensor,
        optimizer: Adam,
        lr_scheduler: Union[ReduceLROnPlateau, ExponentialLR, CosineAnnealingWarmRestarts],
        loss_history: np.ndarray,
        contour_history: list,
    ) -> Tuple[list, np.ndarray]:
        """Execute main optimization loop.

        Per-image early stopping: each sample tracks its own best loss / patience.
        Once a sample's loss has been flat for ``early_stopping_patience`` steps it is
        marked converged — its contour is snapshot-frozen and its gradient zeroed so
        it stops moving, while the other samples keep evolving. The loop ends when all
        samples have converged (or n_epochs is reached).
        """
        try:
            B = contour.shape[0]
            best_loss = torch.full((B,), float("inf"), device=self.device)
            patience = torch.zeros(B, dtype=torch.long, device=self.device)
            converged = torch.zeros(B, dtype=torch.bool, device=self.device)
            frozen = contour.detach().clone()  # per-sample freeze target

            logger.info("Starting contour evolution...")

            for i in tqdm(range(self.n_epochs), desc="Optimizing contour"):
                optimizer.zero_grad()

                loss, batch_loss = self._compute_loss(contour)

                self._backward_and_update(loss, contour, optimizer, converged)
                # ReduceLROnPlateau.step needs the metric; the others don't.
                if isinstance(lr_scheduler, ReduceLROnPlateau):
                    lr_scheduler.step(loss.item())
                else:
                    lr_scheduler.step()

                contour = self._smooth_contour(contour)

                contour_cleaned = self._save_history(
                    contour, batch_loss, loss_history, contour_history, i
                )

                # Per-image freeze: converged samples keep their snapshot; others move.
                if bool(converged.any()):
                    with torch.no_grad():
                        frozen_cleaned = contour_cleaned.detach()
                        frozen_cleaned[converged] = frozen[converged]
                    contour_cleaned = frozen_cleaned.requires_grad_(True)

                optimizer.param_groups[0]["params"][0] = contour_cleaned
                contour = contour_cleaned

                # Per-sample early-stopping bookkeeping, then snapshot+freeze the
                # newly converged samples.
                with torch.no_grad():
                    cur = batch_loss.detach()
                    improved = cur < best_loss - self.early_stopping_threshold
                    best_loss = torch.where(improved, cur, best_loss)
                    patience = torch.where(
                        improved, torch.zeros_like(patience), patience + 1
                    )
                    newly = (~converged) & (patience >= self.early_stopping_patience)
                    if bool(newly.any()):
                        frozen[newly] = contour.detach()[newly]
                        converged = converged | newly

                if bool(converged.all()):
                    logger.info(f"All samples converged (early stop) at epoch {i + 1}")
                    break

            return contour_history, loss_history

        except Exception as e:
            logger.error(f"Error during optimization loop: {e}")
            raise

    def _compute_loss(self, contour: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute loss for current step."""
        try:
            use_amp = (
                self.use_mixed_precision
                and self.device is not None
                and self.device.type == "cuda"
            )
            ctx = torch.amp.autocast('cuda') if use_amp else contextlib.nullcontext()
            with ctx:
                features = self.ctf(contour)
                ms = self.multiscale_loss(features)  # (B,), negative separation energy
                a = area(contour)[:, 0]  # (B,)
                # Scale the area/balloon force by the per-sample separation magnitude
                # (detached) so area_force is a consistent RELATIVE weight regardless
                # of the feature-difference range.
                area_term = self.lambda_area * ms.detach().abs() * a
                batch_loss = ms + area_term
                loss = self.img_dim[0] * torch.mean(batch_loss)
            return loss, batch_loss

        except Exception as e:
            logger.error(f"Error computing loss: {e}")
            raise

    def _backward_and_update(
        self,
        loss: torch.Tensor,
        contour: torch.Tensor,
        optimizer: Adam,
        converged: torch.Tensor = None,
    ) -> None:
        """Perform backward pass and parameter update.

        ``converged`` (B,) bool masks out samples whose contour is frozen: their
        gradient is zeroed so the optimizer leaves them untouched.
        """
        try:
            def _freeze_grad():
                if converged is not None and contour.grad is not None and bool(converged.any()):
                    contour.grad[converged] = 0

            def _add_edge_balloon():
                # Edge-gated outward push added to the gradient (before freezing, so
                # converged samples stay frozen).
                if self._edge_map is not None and contour.grad is not None:
                    if self.edge_balloon > 0:
                        contour.grad = contour.grad + self._edge_balloon_grad(contour)
                    if self.edge_attract > 0:
                        contour.grad = contour.grad + self._edge_attract_grad(contour)

            if self.use_mixed_precision and self.device is not None and self.device.type == "cuda":
                self.scaler.scale(loss).backward(inputs=contour)
                self.scaler.unscale_(optimizer)
                _add_edge_balloon()
                _freeze_grad()
                clip_grad_norm_(contour, self.clip)
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                loss.backward(inputs=contour)
                _add_edge_balloon()
                _freeze_grad()
                clip_grad_norm_(contour, self.clip)
                optimizer.step()

        except Exception as e:
            logger.error(f"Error during backward pass: {e}")
            raise

    def _smooth_contour(self, contour: torch.Tensor) -> torch.Tensor:
        """Apply smoothing to contour."""
        try:
            with torch.no_grad():
                return self.smooth(contour.to(torch.float32))
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

                contour_scaled = contour * self._hist_scale  # cached (H, W) scale
                contour_scaled_contiguous = contour_scaled.contiguous()
                contour_history.append(
                    contour_scaled_contiguous.cpu().detach().numpy().astype(np.int32)
                )

                contour_np = contour.cpu().detach().numpy()
                contour_cleaned_np = self.cleaner.clean_contours_and_interpolate(
                    contour_np
                )
                contour_cleaned = torch.clip(torch.from_numpy(contour_cleaned_np), 0, 1).to(torch.float32).to(self.device)

                contour_cleaned.grad = None
                contour_cleaned.requires_grad = True

                # NOTE: torch.cuda.empty_cache() was called here every epoch, forcing a
                # full device sync + allocator churn ~250x/image. The loop keeps no GPU
                # tensors across epochs, so it is unnecessary — moved to a single call
                # at the end of predict() (~1.23x faster).
                return contour_cleaned

        except Exception as e:
            logger.error(f"Error saving history: {e}")
            raise

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

                    # Knee of the energy curve (see deep_contourflow.knee): end of
                    # the last descent phase. Trim to the real frames first (loss is
                    # padded to n_epochs; only len(contour_history_array) frames exist
                    # after early stop) so the slope detection doesn't see padding.
                    curve = valid_loss[: len(contour_history_array)]
                    index_stop = knee_index(curve)
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
