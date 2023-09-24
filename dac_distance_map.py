from utils import compute_correlogram, delete_loops, augmentation
from scipy.spatial.distance import cdist
from torchvision.models import vgg16
from torch.nn import CosineSimilarity, MSELoss, Module
from torchvision import transforms
import torch.nn.functional as F
from typing import List, Tuple
from torchstain import MacenkoNormalizer
import torch
import cv2
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt

vgg16 = vgg16(pretrained=True)
mse = MSELoss(size_average=None, reduce=None, reduction="mean")
from scipy.interpolate import CubicSpline
from tqdm import tqdm

model = vgg16.features
mse = MSELoss(size_average=None, reduce=None, reduction="mean")



class Isoline_to_features(Module):
    def __init__(self, shapes, isolines: torch.Tensor, vars: torch.Tensor):
        super(Isoline_to_features, self).__init__()

        self.isolines = isolines
        self.vars = vars
        self.shapes = shapes

    def forward(
        self,
        activations: dict,
        distance: torch.Tensor,
        mask: torch.Tensor,
        compute_features_mask=True,
    ):
        isolines = mask * torch.exp(
            -((self.isolines[:, None, None] - distance) ** (2))
            / (self.vars[:, None, None])
        )

        isolines2 = isolines
        x = np.squeeze(isolines2.cpu().detach().numpy())

        isolines0 = F.interpolate(
            isolines2, size=(self.shapes["0"][2], self.shapes["0"][3]), mode="bilinear"
        )
        isolines1 = F.interpolate(
            isolines2, size=(self.shapes["1"][2], self.shapes["1"][3]), mode="bilinear"
        )
        isolines2 = F.interpolate(
            isolines2, size=(self.shapes["2"][2], self.shapes["2"][3]), mode="bilinear"
        )
        isolines3 = F.interpolate(
            isolines2, size=(self.shapes["3"][2], self.shapes["3"][3]), mode="bilinear"
        )
        isolines4 = F.interpolate(
            isolines2, size=(self.shapes["4"][2], self.shapes["4"][3]), mode="bilinear"
        )

        if compute_features_mask:
            mask2 = mask
            mask0 = F.interpolate(
                mask2, size=(self.shapes["0"][2], self.shapes["0"][3]), mode="bilinear"
            )
            mask1 = F.interpolate(
                mask2, size=(self.shapes["1"][2], self.shapes["1"][3]), mode="bilinear"
            )
            mask3 = F.interpolate(
                mask2, size=(self.shapes["3"][2], self.shapes["3"][3]), mode="bilinear"
            )
            mask4 = F.interpolate(
                mask2, size=(self.shapes["4"][2], self.shapes["4"][3]), mode="bilinear"
            )
            masks = [mask0, mask1, mask2, mask3, mask4]

        isolines = [isolines0, isolines1, isolines2, isolines3, isolines4]

        features = []
        features_mask = []

        for i in range(5):
            ind_i = torch.moveaxis(isolines[i], (0, 1), (1, 0))
            f_s_i = (activations[str(i)] * ind_i).sum(dim=[2, 3]) / ind_i.sum(
                dim=[1, 2, 3]
            )[:, None]
            features.append(f_s_i)
            if compute_features_mask:
                features_mask.append(
                    torch.sum(activations[str(i)] * masks[i], dim=(0, 2, 3))
                    / torch.sum(masks[i], (0, 2, 3))
                )
        return features, features_mask


multiscale = model
multiscale.cuda()


preprocess = transforms.Compose(
    [
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


cos = CosineSimilarity(dim=0, eps=1e-08).cuda()


class DAC:
    def __init__(
        self,
        nb_points=100,
        n_epochs=100,
        nb_augment=10,
        model=multiscale,
        dim=512,
        isolines=np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0]),
        sigma=7,
        learning_rate=5e-2,
        clip=1e-1,
        exponential_decay=0.998,
        thresh=1e-2,
        weights=0.9,
    ):
        super(DAC, self).__init__()

        self.nb_points = nb_points
        self.n_epochs = n_epochs
        self.nb_augment = nb_augment
        self.features_anchor = None
        self.features_anchor_mask = None
        self.model = model
        self.activations = {}
        self.isolines = isolines
        self.model[3].register_forward_hook(self.get_activations("0"))
        self.model[8].register_forward_hook(self.get_activations("1"))
        self.model[15].register_forward_hook(self.get_activations("2"))
        self.model[22].register_forward_hook(self.get_activations("3"))
        self.model[29].register_forward_hook(self.get_activations("4"))
        self.learning_rate = learning_rate
        self.clip = clip
        self.shapes = {}
        self.ed = exponential_decay
        self.vars = torch.tensor(
            self.mean_to_var(self.isolines), device="cuda", dtype=torch.float32
        )
        self.isolines = torch.from_numpy(isolines).cuda()
        self.thresh = thresh
        self.gaussian_sigma = sigma
        self.kernel = self.define_kernel()

        self.normalizer = MacenkoNormalizer(backend="numpy")

        self.normalizer.HERef = np.array(
            [
                [0.47262014, 0.17700575],
                [0.79697804, 0.84033483],
                [0.37610664, 0.51235373],
            ]
        )
        self.normalizer.maxCRef = np.array([1.43072807, 0.98501085])

        self.glcm_anchor = None
        self.features_anchor_mask = None
        self.features_mask = None
        # self.correlogram_anchor = None
        self.weights = torch.tensor([weights, (1 - weights)], device="cuda")
        # self.correlogram = None

    def define_kernel(self):
        mil = self.gaussian_sigma * 10 // 2
        filter = np.arange(self.gaussian_sigma * 10) - mil
        x = np.exp((-1 / 2) * (filter**2) / (2 * (self.gaussian_sigma) ** 2))

        return torch.tensor(x / np.sum(x), device="cuda", dtype=torch.float32)

    def convolve(self, x):
        margin = int(100)
        top = x[:margin]
        bot = x[-margin:]
        out = torch.concatenate([bot, x, top]).T[:, None]
        return torch.squeeze(F.conv1d(out, self.kernel[None, None, :], padding="same"))[
            :, margin:-margin
        ]

    def interpolate(self, contour, n):
        try:
            margin = n
            top = contour[:margin]
            bot = contour[-margin:]
            contour_init_new = np.concatenate([bot, contour, top])
            distance = np.cumsum(
                np.sqrt(np.sum(np.diff(contour_init_new, axis=0) ** 2, axis=1))
            )
            distance = np.insert(distance, 0, 0) / distance[-1]

            indices = np.linspace(0, contour_init_new.shape[0] - 1, n).astype(int)
            Cub = CubicSpline(distance[indices], contour_init_new[indices])
            interp_contour = Cub(np.linspace(distance[margin], distance[-margin], n))
        except:
            print(contour.shape)

        return interp_contour

    def mean_to_var(self, isolines):
        mat = cdist(isolines[:, None], isolines[:, None]) ** 2
        masked_a = -np.min(np.ma.masked_equal(mat, 0.0, copy=False), 0) / (
            8 * np.log(0.5)
        )
        return masked_a

    def get_activations(self, name):
        def hook(model, input: Tuple[torch.Tensor], output):
            self.activations[name] = output.to(torch.float32)
            self.shapes[name] = output.shape

        return hook

    def contour_to_distance_map(self, contour):
        eps = 1e-7
        k = 1e4
        contour = torch.unsqueeze(contour, dim=0)
        diff = -self.mesh + contour
        min_diff = torch.min(torch.norm(diff, dim=-1), dim=1)[0]
        min_diff = min_diff.reshape((self.shapes["2"][2], self.shapes["2"][3]))
        roll_diff = torch.roll(diff, -1, dims=1)
        sign = diff * torch.roll(roll_diff, 1, dims=2)
        sign = sign[:, :, 1] - sign[:, :, 0]
        sign = torch.tanh(k * sign)
        norm_diff = torch.clip(torch.norm(diff, dim=2), eps, None)
        norm_roll = torch.clip(torch.norm(roll_diff, dim=2), eps, None)
        scalar_product = torch.sum(diff * roll_diff, dim=2)
        clip = torch.clip(scalar_product / (norm_diff * norm_roll), -1 + eps, 1 - eps)
        angles = torch.arccos(clip)
        torch.pi = torch.acos(torch.zeros(1)).item() * 2
        sum_angles = -torch.sum(sign * angles, dim=1) / (2 * torch.pi)
        out0 = sum_angles.reshape(1, self.shapes["2"][2], self.shapes["2"][3])
        ret = (out0 * min_diff) / torch.max(out0 * min_diff)
        x = np.squeeze(ret.cpu().detach().numpy())
        return ret, torch.unsqueeze(out0, dim=0)

    def fit(self, img, coordinates, augment=True):
        # clip_value = 3
        img, _ = self.normalizer.normalize(img, stains=True)
        img = img / 255
        # HE = HE.reshape(2, img.shape[0], img.shape[1])[0]
        # HE = np.clip(HE,0, clip_value) / clip_value
        mask = cv2.fillPoly(np.zeros(img.shape[:-1]), [coordinates], 1)
        # self.correlogram_anchor = compute_correlogram(HE, mask, len(self.isolines), 15)

        with torch.no_grad():
            if augment == True:
                for i in tqdm(range(self.nb_augment)):
                    img1, mask1 = augmentation(img, mask)
                    img_anchor = img1.astype(np.float32).copy()
                    mask_anchor = mask1.copy()
                    distance_anchor = distance_transform_edt(mask_anchor)
                    distance_anchor = distance_anchor / np.max(distance_anchor)

                    tensor_mask_anchor = torch.tensor(
                        np.transpose(mask_anchor, (-1, 0, 1)), device="cuda"
                    )[None]
                    tensor_anchor = torch.tensor(
                        np.transpose(img_anchor, (-1, 0, 1)), device="cuda"
                    )[None]
                    input_tensor = tensor_anchor
                    x = self.model(preprocess(input_tensor))

                    tensor_distance_anchor = torch.tensor(
                        np.transpose(distance_anchor, (-1, 0, 1)), device="cuda"
                    )[None]
                    tensor_distance_anchor = F.interpolate(
                        tensor_distance_anchor,
                        size=(self.shapes["2"][2], self.shapes["2"][3]),
                        mode="bilinear",
                    )
                    tensor_mask_anchor = F.interpolate(
                        tensor_mask_anchor,
                        size=(self.shapes["2"][2], self.shapes["2"][3]),
                        mode="bilinear",
                    )
                    self.itf = Isoline_to_features(
                        self.shapes, self.isolines, self.vars
                    )

                    if i == 0:
                        features_anchor, self.features_anchor_mask = self.itf(
                            self.activations, tensor_distance_anchor, tensor_mask_anchor
                        )
                    else:
                        tmp, tmp_mask = self.itf(
                            self.activations, tensor_distance_anchor, tensor_mask_anchor
                        )

                        for j, u in enumerate(zip(tmp, tmp_mask)):
                            features_anchor[j] += u[0]
                            self.features_anchor_mask[j] += u[1]

                        del tmp

                    del (
                        x,
                        tensor_distance_anchor,
                        tensor_anchor,
                        tensor_mask_anchor,
                        input_tensor,
                    )

                self.features_anchor = [u / self.nb_augment for u in features_anchor]
                self.features_anchor_mask = [
                    u / self.nb_augment for u in self.features_anchor_mask
                ]

    def forward_on_epoch(self, contour):
        dst_map, mask = self.contour_to_distance_map(contour)

        features, self.features_mask = self.itf(
            self.activations, dst_map, mask, compute_features_mask=False
        )

        #### Initialize variables
        tot = torch.zeros(len(self.shapes), device="cuda")
        score = torch.zeros(len(self.shapes), device="cuda")
        energies = torch.zeros((len(self.shapes), len(self.isolines)))
        arr0 = torch.tensor(
            [1.0, 1 / 2, 1 / 4.0, 1 / 8.0, 1 / 16.0], device="cuda", dtype=torch.float32
        )
        arr = arr0 / torch.sum(arr0)

        #### compute loss

        for j, feature in enumerate(features):
            difference_features = feature - self.features_anchor[j]
            tmp = torch.sqrt(torch.norm(difference_features, dim=1))
            energies[j] = tmp
            tot[j] = arr[j] * torch.mean(self.weights * tmp)

        tot = tot.flatten().mean(-1)

        return tot, energies

    def predict(self, img, contour_init):

        # clip_value = 3
        img, _ = self.normalizer.normalize(img, stains=True)
        img = img / 255
        # HE = HE.reshape(2, img.shape[0], img.shape[1])[0]
        # HE = np.clip(HE,0, clip_value) / clip_value
        
        #### Initialize variables
        self.dims = np.array(img.shape[:-1])

        tots = np.zeros(self.n_epochs)
        contours = np.zeros((self.n_epochs, self.nb_points, 2))
        energies = torch.zeros((self.n_epochs, len(self.shapes), len(self.isolines)))
        scale = torch.tensor(
            [512.0, 512.0], device="cuda", dtype=torch.float32
        ) / torch.tensor(self.dims.copy(), device="cuda", dtype=torch.float32)
        stop = False
        contour_init = contour_init / np.flip(self.dims)
        contour_init = self.interpolate(contour_init, self.nb_points)
        contour_init = np.roll(contour_init, axis=1, shift=1)

        #### pass image into neural network and get features
        tensor = torch.tensor(
            np.transpose(img.astype(np.float32), (-1, 0, 1)).copy(), device="cuda"
        )
        input_tensor = torch.unsqueeze(tensor, 0)
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
        self.itf = Isoline_to_features(self.shapes, self.isolines, self.vars)
        energies = np.zeros((self.n_epochs, len(self.shapes), len(self.isolines)))

        contour = torch.from_numpy(contour_init.astype(np.float32)).cuda()
        contour.requires_grad = True

        for i in range(self.n_epochs):
            tot, energie = self.forward_on_epoch(contour)
            tots[i] = tot
            contours[i] = contour.cpu().detach().numpy()
            energies[i] = energie.cpu().detach().numpy()

            tot.backward(inputs=contour)

            norm_grad = torch.unsqueeze(torch.norm(contour.grad, dim=1), -1)
            clipped_norm = torch.clip(norm_grad, 0, self.clip)
            stop = torch.max(norm_grad) < self.thresh

            if stop == False:
                with torch.no_grad():
                    gradient_direction = contour.grad * clipped_norm / norm_grad
                    gradient_direction = self.convolve(gradient_direction).T
                    contour = (
                        contour
                        - scale
                        * self.learning_rate
                        * (self.ed**i)
                        * gradient_direction
                    )
                    contour = contour.cpu().detach().numpy()
                    interpolated_contour = self.interpolate(
                        contour, n=self.nb_points
                    ).astype(np.float32)
                contour = delete_loops(contour,self.dims)
                contour = torch.from_numpy(interpolated_contour).cuda()
                # contour = delete_loops(contour,self.dims)
                contour.grad = None
                contour.requires_grad = True

            else:
                print("the algorithm stoped earlier")
                break

        ##### Calculate score after gradient descent
        dst_map, mask = self.contour_to_distance_map(contour)
        _, self.features_mask = self.itf(self.activations, dst_map, mask)

        score = np.zeros(len(self.shapes))

        for i in range(len(self.shapes)):
            score[i] = (
                cos(self.features_anchor_mask[i], self.features_mask[i])
                .cpu()
                .detach()
                .numpy()
            )

        contours = np.roll(contours, 1, -1)
        contours = (contours * np.flip(self.dims)).astype(np.int32)

        tots[tots == 0] = 1e10
        argmin = np.argmin(tots)

        mask = np.zeros(img.shape[:-1])
        mask = cv2.fillPoly(mask, [contours[argmin].astype(int)], 1)

        # self.correlogram = compute_correlogram(HE, mask, len(self.isolines), 15)
        # score_correlogram = np.mean(
        #     np.sum(np.minimum(self.correlogram, self.correlogram_anchor), axis=-1)
        #     / np.sum(np.maximum(self.correlogram, self.correlogram_anchor), axis=-1)
        # )
        # score = np.append(score, [score_correlogram])

        return contours, score, tots, energies
