import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from .utils_dilated_tubules import *
from torchvision import models, transforms
import torchstain
from tqdm import tqdm
from utils import *
from typing import List, Tuple

preprocess = transforms.Compose(
    [
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
vgg16 = models.vgg16(pretrained=True)
VGG16 = vgg16.features


class DCF:
    def __init__(
        self,
        n_epochs=100,
        nb_augment=100,
        model=VGG16,
        isolines=np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1.0]),
        sigma=7,
        learning_rate=5e-2,
        clip=1e-1,
        exponential_decay=0.998,
        thresh=1e-2,
        weights=0.9,
    ):
        super(DCF, self).__init__()

        self.n_epochs = n_epochs
        self.nb_augment = nb_augment
        self.model = model
        self.activations = {}
        self.isolines = torch.from_numpy(isolines)
        self.model[3].register_forward_hook(self.get_activations(0))
        self.model[8].register_forward_hook(self.get_activations(1))
        self.model[15].register_forward_hook(self.get_activations(2))
        self.model[22].register_forward_hook(self.get_activations(3))
        self.model[29].register_forward_hook(self.get_activations(4))
        self.learning_rate = learning_rate
        self.clip = clip
        self.ed = exponential_decay
        self.thresh = thresh
        self.smooth = Smoothing(sigma)
        self.ctf = Contour_to_isoline_features(
            256, halfway_value=0.5, isolines=self.isolines
        )
        self.dtf = Distance_map_to_isoline_features(self.isolines, halfway_value=0.5)
        self.normalizer = torchstain.normalizers.MacenkoNormalizer(backend="torch")
        self.normalizer.HERef = np.array(
            [
                [0.47262014, 0.17700575],
                [0.79697804, 0.84033483],
                [0.37610664, 0.51235373],
            ]
        )
        self.normalizer.maxCRef = np.array([1.43072807, 0.98501085])
        self.isoline_weights = torch.tensor([weights, (1 - weights)])
        self.cleaner = CleanContours()

    def get_activations(self, name: str):
        """
        Returns a hook function that stores the activations (outputs) of a layer in a dictionary
        under the given name. This hook is designed to be registered on a specific layer in a model,
        allowing you to capture its output (activations) during the forward pass.

        Parameters:
        -----------
        name : str
             A string that identifies the name/key under which the activations of the layer
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

            device = input[0].device
            self.activations[name] = output.to(device)

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
                          A list of feature isoline tensors for the query, where each tensor corresponds to a layer’s
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
        batch_size = features_isolines[0].shape[0]
        loss_scales = torch.zeros((batch_size, len(self.activations)))
        loss_scales_isos = torch.zeros(
            (batch_size, len(self.activations), len(self.isolines))
        )

        if str(self.device) == "cuda:0":
            loss_scales, self.isoline_weights = (
                loss_scales.cuda(),
                self.isoline_weights.cuda(),
            )
        for j in range(len(features_isolines)):
            difference_features = (
                features_isolines[j] - self.features_isolines_support[j]
            )
            lsi = torch.sqrt(torch.norm(difference_features, dim=-2))

            loss_scales_isos[:, j], loss_scales[:, j] = lsi, torch.mean(
                self.isoline_weights * lsi, dim=-1
            )
        loss_batch = torch.mean(loss_scales, dim=1)
        loss_scales_isos_batch = torch.mean(loss_scales_isos, dim=1)
        return loss_batch, loss_scales_isos_batch

    def fit(self, img_support, polygon_support, augment=True):
        self.device = img_support.device
        if img_support.dtype != torch.float32:
            raise Exception("tensor must be of type float32")

        # img_support, _, _ = self.normalizer.normalize(
        #     (img_support * 255)[0].to(torch.int32), stains=True
        # )
        # img_support_array = img_support_array / 255

        ctd = Contour_to_distance_map(size=512)
        distance_map_support, mask_support = ctd(polygon_support, return_mask=True)

        with torch.no_grad():
            if augment == False:
                self.nb_augment = 1

            else:
                for i in tqdm(range(self.nb_augment)):
                    img_augmented, mask_augmented, distance_map_support_augmented = (
                        augmentation(img_support, mask_support, distance_map_support)
                    )
                    if str(self.device) == "cuda:0":
                        self.model = self.model.cuda()
                        _ = self.model(preprocess(img_augmented))

                    if i == 0:
                        self.features_isolines_support, self.features_mask_support = (
                            self.dtf(
                                self.activations,
                                distance_map_support_augmented,
                                mask_augmented,
                                compute_features_mask=True,
                            )
                        )

                    else:
                        tmp, tmp_mask = self.dtf(
                            self.activations,
                            distance_map_support_augmented,
                            mask_augmented,
                            compute_features_mask=True,
                        )

                        for j, u in enumerate(zip(tmp, tmp_mask)):
                            self.features_isolines_support[j] += u[0]
                            self.features_mask_support[j] += u[1]

                self.features_isolines_support = [
                    u / self.nb_augment for u in self.features_isolines_support
                ]
                self.features_anchor_mask = [
                    u / self.nb_augment for u in self.features_mask_support
                ]
        self.weights = torch.tensor(
            [1 / (2) ** i for i in range(len(self.activations))],
            dtype=torch.float32,
        )
        self.weights = self.weights / torch.sum(self.weights)

        if str(self.device) == "cuda:0":
            self.weights = self.weights.cuda()

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

        b = features_mask_query[0].shape[0]
        score = torch.zeros((len(self.activations), b))
        for i in range(len(self.activations)):

            cos = (
                self.weights[i]
                * self.features_mask_support[i]
                @ features_mask_query[i].T.to(torch.float32)
                / (
                    torch.linalg.norm(self.features_mask_support[i])
                    * torch.linalg.norm(features_mask_query[i]).to(torch.float32)
                )
            )

            score[i] = torch.flatten(cos)
        return torch.mean(score, dim=0)

    def predict(
        self, imgs_query, contours_query
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
                  - H the Heigth of the images
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
        self.nb_points = contours_query.shape[-2]
        batch_size, _, h, w = imgs_query.shape
        stop = torch.zeros(batch_size, dtype=torch.bool)
        losses = np.zeros((self.n_epochs, batch_size))

        epochs_contours_query = np.zeros(
            (self.n_epochs, batch_size, contours_query.shape[-2], 2)
        )
        loss_scales_isos = np.zeros(
            (self.n_epochs, batch_size, len(self.activations), len(self.isolines))
        )
        scale = torch.tensor(
            [512.0, 512.0], device="cuda", dtype=torch.float32
        ) / torch.tensor([h, w].copy(), device="cuda", dtype=torch.float32)

        # for i in range(batch_size):
        #     normalized_img, _, _ = self.normalizer.normalize((imgs_query*255).to(torch.int)[i], stains=True)
        #     imgs_query[i] = normalized_img/255

        contours_query_array = contours_query.cpu().detach().numpy()
        contours_query_array = contours_query_array / np.flip([h, w])
        contours_query_array = self.cleaner.clean_contours_and_interpolate(
            contours_query_array
        )
        contours_query = torch.from_numpy(
            np.roll(contours_query_array, axis=1, shift=1)
        )
        contours_query.requires_grad = True

        #### pass image into neural network and get features
        if str(self.device) == "cuda:0":
            self.model = self.model.cuda()
            _ = self.model(preprocess(imgs_query))
            contours_query = contours_query.cuda()

        #### Begin the gradient descent
        for i in range(self.n_epochs):
            features_isoline_query, _ = self.ctf(contours_query, self.activations)
            loss_batch, loss_scales_isos_batch = self.multi_scale_multi_isoline_loss(
                features_isoline_query
            )
            loss_all = torch.mean(loss_batch)
            loss_batch[~stop].backward(inputs=contours_query)
            losses[i] = loss_all.cpu().detach().numpy()
            epochs_contours_query[i] = contours_query.cpu().detach().numpy()
            loss_scales_isos[i] = (
                torch.mean(loss_scales_isos_batch, dim=0).cpu().detach().numpy()
            )
            norm_grad = torch.unsqueeze(torch.norm(contours_query.grad, dim=-1), -1)
            clipped_norm = torch.clip(norm_grad, 0, self.clip)
            stop = (torch.amax(norm_grad[:, 0], dim=-2) < self.thresh)[-1]
            if (
                torch.all(stop) != True
            ):  #### if the contour is not moving much anymore just stop the gradient descent

                with torch.no_grad():
                    gradient_direction = contours_query.grad * clipped_norm / norm_grad
                    gradient_direction = self.smooth(
                        gradient_direction.to(torch.float32)
                    )
                    contours_query = (
                        contours_query
                        - scale * self.learning_rate * (self.ed**i) * gradient_direction
                    )
                interpolated_contour = self.cleaner.clean_contours_and_interpolate(
                    contours_query.cpu().detach().numpy()
                )
                contours_query = torch.from_numpy(interpolated_contour).cuda()
                contours_query.grad = None
                contours_query.requires_grad = True

            else:
                print("the algorithm stoped earlier")
                break

        ##### Calculate score after gradient descent
        self.ctf.compute_features_mask = True
        _, features_mask_query = self.ctf(contours_query, self.activations)
        scores = self.similarity_score(features_mask_query).cpu().detach().numpy()

        contours_query = np.roll(contours_query.cpu().detach().numpy(), 1, -1)
        contours_query = (contours_query * np.flip(imgs_query.shape[-2:])).astype(
            np.int32
        )

        losses[losses == 0] = 1e10
        argmin = np.argmin(losses, axis=0)

        return epochs_contours_query[argmin], scores, losses, loss_scales_isos
