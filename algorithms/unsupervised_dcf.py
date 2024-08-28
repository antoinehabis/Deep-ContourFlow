import sys
from pathlib import Path
from utils import *

sys.path.append(str(Path(__file__).resolve().parent.parent))
import torchvision.models as models
from torch.nn import MSELoss
from torchvision import transforms
from torch_contour.torch_contour import area, Smoothing, CleanContours
import torch
from tqdm import tqdm


preprocess = transforms.Compose(
    [
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
mse = MSELoss(size_average=None, reduce=None, reduction="mean")
vgg16 = models.vgg16(pretrained=True)
VGG16 = vgg16.features


class DCF:
    def __init__(
        self,
        n_epochs=100,
        model=VGG16,
        learning_rate=5e-2,
        clip=1e-1,
        exponential_decay=0.998,
        thresh=1e-2,
        weights=[1, 1, 1, 1, 1],
        area_force=0.0,
        sigma=1,
    ):
        super(DCF, self).__init__()

        self.n_epochs = n_epochs
        self.model = model
        self.activations = {}

        self.model[3].register_forward_hook(self.get_activations(0))
        self.model[8].register_forward_hook(self.get_activations(1))
        self.model[15].register_forward_hook(self.get_activations(2))
        self.model[22].register_forward_hook(self.get_activations(3))
        self.model[29].register_forward_hook(self.get_activations(4))

        self.learning_rate = learning_rate
        self.clip = clip
        self.shapes = {}
        self.ed = exponential_decay
        self.thresh = thresh
        self.lambda_area = area_force
        self.weights = weights / np.sum(weights)
        self.smooth = Smoothing(sigma)
        self.cleaner = CleanContours()
        self.device = None
        self.Ctf = Contour_to_features(256)
    def get_activations(self, name):
        def hook(model, input, output):

            device = input[0].device
            self.activations[name] = output.to(device)

        return hook

    def multiscale_loss(self, features, weights, eps = 1e-2):
        """
        Computes a multiscale loss based on the features inside and outside the mask and given weights.

        This method calculates the energy difference between features inside and outside the mask for each scale,
        normalizes it by the mean activation, and then sums these energies weighted by the given weights.

        Parameters:
        -----------
        features : tuple of (list of torch.Tensor, list of torch.Tensor)
            A tuple containing two lists: features_inside and features_outside for each scale.
        weights : list or numpy array
            A list or numpy array of weights for each scale.

        Returns:
        --------
        torch.Tensor
            The computed multiscale loss.
        """
        
        features_inside, features_outside = features
        nb_scales = len(features_inside)
        weights = torch.tensor(weights, device=self.device).requires_grad_(False)
        energies = torch.zeros(nb_scales, device=self.device)

        for j in range(len(features_inside)):
            tmp = -torch.linalg.vector_norm(
                features_inside[j] - features_outside[j], 2, dim=-1
            ) / (
                torch.linalg.vector_norm(self.activations[j], 2, dim=(1, 2, 3))
                + eps
            )
            energies[j] = torch.mean(tmp)

        fin = torch.sum(energies * weights)
        return fin

    def predict(self, img, contour_init):
        """
        Predicts the contour for a given image and initial contour using a specified model.

        This method performs a series of operations including preprocessing the image, initializing the contour,
        and iteratively updating the contour using gradient-based optimization until a stopping criterion is met.

        Parameters:
        -----------
        img : torch.Tensor
            The input image tensor of shape (B, C, H, W) where B is the batch size, C is the number of channels, H is the height, and W is the width.
            The image tensor must be of type float32.
        contour_init : torch.Tensor
            The initial contour tensor.
            the initial contour must be of shape (B, 1, K, 2) where B is the batch size, K is the number of node.

        Returns:
        --------
        contour_history : np.ndarray
            A numpy array containing the history of contours during the prediction process.
            contour_history has shape(self.n_epochs, B, 1, K, 2)
        loss_history : np.ndarray
            A numpy array containing the loss values during each epoch of the prediction process.

        Raises:
        -------
        Exception
            If the image tensor is not of type float32.
        """

        self.device = contour_init.device
        self.img_dim = torch.tensor(img.shape[-2:], device=self.device)
        size = img.shape[-1] // (2**2)
        stop = False
        loss_history = np.zeros((self.n_epochs))
        contour_history = []

        if img.dtype != torch.float32:
            raise Exception("Image must be of type float32")

        scale = (
            torch.tensor([512.0, 512.0], device=self.device, dtype=torch.float32)
            / self.img_dim
        )
        if str(self.device) == "cuda:0":
            self.model = self.model.cuda()
        _ = self.model(preprocess(img))
        contour = torch.roll(contour_init, dims=-1, shifts=1)

        contour.requires_grad_()

        print('Contour is evolving please wait a few moment...')

        for i in tqdm(range(self.n_epochs)):
            features = self.Ctf(contour, self.activations)
            loss = self.multiscale_loss(features, self.weights) + torch.mean(
                self.lambda_area * area(contour)
            )
            loss_history[i] = loss
            loss.backward(inputs=contour)
            norm_grad = torch.linalg.vector_norm(contour.grad, dim=-1, keepdim=True)
            stop = torch.max(norm_grad) < self.thresh
            contour_history.append(contour.cpu().detach().numpy())
            if stop == False:
                with torch.no_grad():
                    clipped_norm = torch.clamp(norm_grad, 0.0, self.clip)
                    gradient_direction = (
                        contour.grad * clipped_norm / (norm_grad + 1e-8)
                    )
                    gradient_direction = self.smooth(
                        gradient_direction.to(torch.float32)
                    )

                    contour = (
                        contour
                        - scale * self.learning_rate * (self.ed**i) * gradient_direction
                    )
                    contour_without_loops = self.cleaner.clean_contours_and_interpolate(
                        contour.cpu().detach().numpy()
                    )

                contour = torch.clip(torch.from_numpy(contour_without_loops), 0, 1)
                if str(self.device) == "cuda:0":
                    contour = contour.cuda()
                contour.grad = None
                contour.requires_grad_()
            else:
                print("The stopping criterion was reached")
                break

        contour_history = np.roll(np.stack(contour_history), axis=-1, shift=-1)
        contour_history = (
            contour_history * np.array(img.shape[-2:])[None, None, None, None]
        ).astype(np.int32)
        print("Contour stopped")
        return contour_history, loss_history
