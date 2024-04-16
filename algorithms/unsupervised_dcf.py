import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils import *
from scipy.ndimage.morphology import distance_transform_edt
from scipy.spatial.distance import cdist
import torchvision.models as models
from torch.nn import MSELoss, Module
from torchvision import transforms
import torch.nn.functional as F
from typing import List, Tuple
import torch
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from torch import autograd

vgg16 = models.vgg16(pretrained=True)
model = vgg16.features
model.cuda()

mse = MSELoss(size_average=None, reduce=None, reduction="mean")


class Mask_to_features(Module):
    def __init__(
        self,
        shapes,
    ):
        super(Mask_to_features, self).__init__()
        self.shapes = shapes

    def forward(self, activations: dict, mask: torch.tensor):
        mask0 = F.interpolate(
            mask, size=(self.shapes["0"][2], self.shapes["0"][3]), mode="bilinear"
        )
        mask1 = F.interpolate(
            mask, size=(self.shapes["1"][2], self.shapes["1"][3]), mode="bilinear"
        )
        mask2 = mask
        mask3 = F.interpolate(
            mask, size=(self.shapes["3"][2], self.shapes["3"][3]), mode="bilinear"
        )
        mask4 = F.interpolate(
            mask, size=(self.shapes["4"][2], self.shapes["4"][3]), mode="bilinear"
        )

        masks = [mask0, mask1, mask2, mask3, mask4]

        features_inside, features_outside = [], []
        for i in range(5):
            features_inside.append(
                torch.sum(activations[str(i)] * masks[i], dim=(0, 2, 3))
                / (torch.sum(masks[i], (0, 2, 3)) + 1e-5)
            )

            features_outside.append(
                torch.sum(activations[str(i)] * (1 - masks[i]), dim=(0, 2, 3))
                / (torch.sum((1 - masks[i]), (0, 2, 3)) + 1e-5)
            )



        return features_inside, features_outside


multiscale = vgg16.features
multiscale.cuda()


preprocess = transforms.Compose(
    [
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class DCF:
    def __init__(
        self,
        nb_points=100,
        n_epochs=100,
        model=multiscale,
        learning_rate=5e-2,
        clip=1e-1,
        exponential_decay=0.998,
        thresh=1e-2,
        weights = [1,1,1,1,1],
        area_force=0.,
        gaussian_sigma = 1
    ):
        super(DCF, self).__init__()

        self.nb_points = nb_points
        self.n_epochs = n_epochs
        self.model = model
        self.activations = {}
        self.model[3].register_forward_hook(self.get_activations("0"))
        self.model[8].register_forward_hook(self.get_activations("1"))
        self.model[15].register_forward_hook(self.get_activations("2"))
        self.model[22].register_forward_hook(self.get_activations("3"))
        self.model[29].register_forward_hook(self.get_activations("4"))
        self.learning_rate = learning_rate
        self.clip = clip
        self.shapes = {}
        self.ed = exponential_decay
        self.thresh = thresh
        self.gaussian_sigma = gaussian_sigma
        self.kernel = self.define_kernel()
        self.lambda_area = area_force
        self.weights = weights
    
    def define_kernel(self):
        mil = self.gaussian_sigma * 10 * 1 // 2
        filter = np.arange(self.gaussian_sigma * 10) - mil
        x = np.exp((-1 / 2) * (filter**2) / (2 * (self.gaussian_sigma) ** 2))

        return torch.tensor(x / np.sum(x), device="cuda", dtype=torch.float32)

    def convolve(self, x):
        margin = x.shape[0]//2
        top = x[:margin]
        bot = x[-margin:]
        out = torch.concatenate([bot, x, top]).T[:, None]
        return torch.squeeze(F.conv1d(out, self.kernel[None, None, :], padding="same"))[
            :, margin:-margin
        ]

    def interpolate(self, contour, n):
        margin = n // 10
        top = contour[:margin]
        bot = contour[-margin:]
        contour_init_new = np.concatenate([bot, contour, top])
        distance = np.cumsum(
            np.sqrt(np.sum(np.diff(contour_init_new, axis=0) ** 2, axis=1))
        )
        distance = np.insert(distance, 0, 0) / distance[-1]
        boolean_equal = (np.roll(np.diff(distance,append=2),1)==0).astype(bool)
        distance[boolean_equal] = distance[boolean_equal] + 1e-6

        indices = np.linspace(0, contour_init_new.shape[0] - 1, 100).astype(int)
        indices = np.unique(indices)

        Cub = CubicSpline(distance[indices], contour_init_new[indices])
        interp_contour = Cub(np.linspace(distance[margin], distance[-margin], n))

        return interp_contour

    def get_activations(self, name):
        def hook(model, input: Tuple[torch.Tensor], output):
            self.activations[name] = output.to(torch.float32)
            self.shapes[name] = output.shape

        return hook

    def area_force(self, contour):
        x, y  = contour[:,0], contour[:,1]
        area = -0.5*torch.abs(torch.dot(x,torch.roll(y,1))-torch.dot(y,torch.roll(x,1)))
        return area

    def contour_to_mask(self, contour):
        k = 1e5
        eps = 1e-6
        contours = torch.unsqueeze(contour, dim=0)
        diff = -self.mesh + contours
        roll_diff = torch.roll(diff, -1, dims=1)
        sign = diff * torch.roll(roll_diff, 1, dims=2)
        sign = sign[:, :, 1] - sign[:, :, 0]
        sign = torch.tanh(k * sign)
        norm_diff = torch.linalg.vector_norm(diff, dim=2)
        norm_roll = torch.linalg.vector_norm(roll_diff, dim=2)
        scalar_product = torch.sum(diff * roll_diff, dim=2)
        clip = torch.clamp(scalar_product / (norm_diff * norm_roll ),-1 + eps, 1 - eps)
        angles = torch.acos(clip)
        torch.pi = torch.acos(torch.zeros(1)).item() * 2
        sum_angles = torch.clamp(-torch.sum(sign * angles, dim=1) / (2 * torch.pi), 0, 1)
        out0 = sum_angles.reshape(1, self.shapes["2"][2], self.shapes["2"][3])
        mask = torch.unsqueeze(out0, dim=0)
        area_force = self.area_force(contour)
        return mask, area_force

    def forward_on_epoch(self, contour,epoch):
        mask, area_force = self.contour_to_mask(contour)
        features_inside, features_outside = self.mtf(self.activations, mask)

        arr0 = torch.tensor(self.weights, device="cuda")
        energies = torch.zeros(len(self.shapes)).cuda()
        arr = arr0 / torch.sum(arr0)
        for j in range(len(features_inside)):
            energies[j] = -torch.linalg.vector_norm(torch.clamp((features_inside[j] - features_outside[j])/(torch.mean(self.activations[str(j)], dim=(0, 2, 3))+1e-6),1e-6),2)
        fin = torch.sum(energies * arr0)
        return fin + self.lambda_area*area_force

    def predict(self, img, contour_init):
        img = img / 255
        self.dims = np.array(np.flip(img.shape[:-1]))
        scale = torch.tensor(
            [512.0, 512.0], device="cuda", dtype=torch.float32
        ) / torch.tensor(self.dims.copy(), device="cuda", dtype=torch.float32)
        img_anchor = img.astype(np.float32).copy()
        tensor_anchor = torch.unsqueeze(
            torch.tensor(np.transpose(img_anchor, (-1, 0, 1)), device="cuda"), 0
        )
        input_tensor = tensor_anchor

        x = self.model(preprocess(input_tensor))

        self.mesh = torch.unsqueeze(
            torch.stack(
                torch.meshgrid(
                    torch.arange(self.shapes["2"][2]), torch.arange(self.shapes["2"][3])
                ),
                dim=-1,
            ).reshape(-1, 2),
            dim=1,
        )

        self.mesh = self.mesh.to(torch.float32).cuda() / torch.tensor(
            self.shapes["2"][2:], dtype=torch.float32, device="cuda"
        )
        self.mtf = Mask_to_features(self.shapes)

        if img.dtype != "float64":
            raise Exception("Image must be normalized between 0 and 1")

        stop = False
        tots = np.zeros((self.n_epochs))
        scores = np.zeros((self.n_epochs, len(self.shapes)))
        contours = []
        contour_init = contour_init / self.dims
        tensor = torch.tensor(
            np.transpose(img.astype(np.float32), (-1, 0, 1)).copy(), device="cuda"
        )
        input_tensor = torch.unsqueeze(tensor, 0)
        x = self.model(preprocess(input_tensor))

        contour_init = self.interpolate(contour_init, self.nb_points)
        contour_init = np.roll(contour_init, axis=1, shift=1)

        contour = torch.from_numpy(contour_init).cuda()
        contour.requires_grad = True

        for i in range(self.n_epochs):
            with autograd.detect_anomaly():

                tot = self.forward_on_epoch(contour,i+1)
                tots[i] = tot
                tot.backward(inputs=contour)

            norm_grad = torch.unsqueeze(torch.linalg.vector_norm(contour.grad, dim=1), -1)
            stop = torch.max(norm_grad) < self.thresh
            contours.append(contour.cpu().detach().numpy())
            if stop == False:
                with torch.no_grad():
                    clipped_norm = torch.clamp(norm_grad, 0., self.clip)
                    gradient_direction = contour.grad * clipped_norm / (norm_grad +1e-5)
                    gradient_direction = self.convolve(
                        gradient_direction.to(torch.float32)
                    ).T
                    contour = (
                        contour
                        - scale
                        * self.learning_rate
                        * (self.ed**i)
                        * gradient_direction
                    )
                    contour = contour.cpu().detach().numpy()
                    try:
                        contour_without_loops = delete_loops(contour)
                    except:
                        contour_without_loops = contour
                    if contour_without_loops.shape[0] == 0:
                        interpolated_contour = self.interpolate(
                            contour, n=self.nb_points
                        ).astype(np.float32)

                    else:
                        interpolated_contour = self.interpolate(
                            contour_without_loops, n=self.nb_points
                        ).astype(np.float32)

                contour = torch.clip(
                    torch.from_numpy(interpolated_contour).cuda(), 0, 1
                )
                contour.grad = None
                contour.requires_grad = True
            else:
                print("the algorithm stopped earlier")
                break

        contours = np.roll(np.array(contours), 1, -1)
        contours = (contours * self.dims).astype(np.int32)
        return contours, tots
