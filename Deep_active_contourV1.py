from utils import *
from augmentation import *
from config import *
from scipy.ndimage.morphology import distance_transform_edt
from scipy.spatial.distance import cdist
import torchvision.models as models
from torch.nn import AvgPool2d, MaxPool2d, CosineSimilarity, MSELoss, Module, ReLU, Linear, Sigmoid
from torchvision import transforms
import torch.nn.functional as F
from typing import List, Tuple
vgg16 = models.vgg16(pretrained=True)
model = vgg16.features
model.cuda()

mse = MSELoss(size_average=None, reduce=None, reduction='mean')

    
class Mask_to_features(Module):

    def __init__(self):

        super(Mask_to_features, self).__init__()


    def forward(self,
                activations: dict,
                mask: torch.tensor):
        

        mask0 = F.interpolate(mask, size=(512,512), mode = 'nearest') 
        mask1 = F.interpolate(mask, size=(256,256), mode = 'nearest')
        mask2 = mask   #### 1x128x128
        mask3 = F.interpolate(mask, size=(64,64), mode = 'nearest')
        mask4 = F.interpolate(mask, size=(32,32), mode = 'nearest') 

        dic = {0:mask0,
               1:mask1,
               2:mask2,
               3:mask3,
               4:mask4}

        features_inside = []
        features_outside = []

        for i in range(5):
   
            features_inside.append(torch.sum(activations[str(i)] * dic[i],dim=(0,2,3))/torch.sum(dic[i],(0,2,3)))
            features_outside.append(torch.sum(activations[str(i)] * (1 - dic[i]),dim=(0,2,3))/torch.sum((1 - dic[i]),(0,2,3)))


        return features_inside, features_outside 

multiscale = vgg16.features
multiscale.cuda()


preprocess = transforms.Compose([
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])





class DAC():
    def __init__(self,
                 nb_points = 100,
                 n_epochs = 100,
                 model = multiscale,
                 dim = 512,
                 learning_rate = 5e-2,
                 clip = 1e-1,
                 exponential_decay = 0.998,
                 thresh = 1e-2):
        
    
        super(DAC, self).__init__() 
        
        self.nb_points = nb_points
        self.n_epochs = n_epochs
        self.model = model
        self.activations = {}
        self.model[3].register_forward_hook(self.get_activations('0'))
        self.model[8].register_forward_hook(self.get_activations('1'))
        self.model[15].register_forward_hook(self.get_activations('2'))
        self.model[22].register_forward_hook(self.get_activations('3'))
        self.model[29].register_forward_hook(self.get_activations('4')) 
        self.dim = dim
        self.learning_rate = learning_rate
        self.clip = clip
        self.shapes = {}
        self.ed = exponential_decay
        self.thresh = thresh
        self.mesh =  torch.unsqueeze(torch.stack(torch.meshgrid(torch.arange(self.dim//4),torch.arange(self.dim//4)),dim = -1).reshape(-1,2)/(self.dim//4), dim=1).cuda()
        self.dst = Mask_to_features()
        self.kernel_size = 10
        self.gaussian_sigma = 1
        self.kernel = self.define_kernel()



    def define_kernel(self):

        mil = self.kernel_size//2
        filter = np.arange(self.kernel_size) - mil
        x = np.exp((-1/2)*(filter**2)/(2*(self.gaussian_sigma)**2))

        return torch.tensor(x/np.sum(x), dtype=float, device='cuda')
    
    def convolve(self, 
                 x,
                 kernel):

        margin = int(30)
        top = x[:margin]
        bot = x[-margin:]
        out = torch.concatenate([bot,x,top]).T[:,None]
        return torch.squeeze(F.conv1d(out, kernel[None,None,:],padding='same'))[:,margin:-margin]

    def interpolate(self,
                    shape,
                    n,
                    margin = 20):
        
        top = shape[:margin]
        bot = shape[-margin:]
        new_shape = np.concatenate([bot,shape,top])
        distance = np.cumsum( np.sqrt(np.sum( np.diff(new_shape, axis=0)**2, axis=1 )) )
        distance = np.insert(distance, 0, 0)/distance[-1]
        alpha = np.linspace(distance[margin], distance[-margin], n)

        out =  interp1d(distance, new_shape, kind='linear', axis=0)(alpha)
        
        return out
    
    def get_activations(self,
                        name):
        def hook(model,
                 input: Tuple[torch.Tensor],
                 output):
            
            self.activations[name] = output.to(torch.float16)
            self.shapes[name] = output.shape
            
        return hook
    
    def contour_to_mask(self,
                        shape):   

            eps = 1e-6
            k = 1e5
            contours = torch.unsqueeze(shape, dim = 0)
            diff = - self.mesh + contours        
            tmp = torch.roll(diff, -1, dims=1)
            sign = diff * torch.roll(tmp,1,dims=2)
            sign = sign[:,:,1] - sign[:,:,0]
            sign = torch.tanh(k * sign)
            norm_tmp = torch.clip(torch.norm(diff, dim = 2), eps, None)
            norm_tmp1 = torch.clip(torch.norm(tmp, dim = 2), eps, None)
            scalar_product= torch.sum(diff * tmp, dim = 2)
            clip = torch.clip(scalar_product/(norm_tmp*norm_tmp1),-1 +eps, 1-eps)
            angles = torch.arccos(clip)
            torch.pi = torch.acos(torch.zeros(1)).item() * 2
            sum_angles = -torch.sum(sign * angles, dim=1)/(2*torch.pi)
            out0 = sum_angles.reshape(1,self.shapes['2'][2], self.shapes['2'][3])
            return torch.unsqueeze(out0, dim = 0)



    
    def forward_on_epoch(self,
                         shape, 
                         i): 
        
        frac = 1-i/ self.n_epochs

        mask = self.contour_to_mask(shape)

        features_inside, features_outside = self.dst(self.activations,
                                                     mask)
        arr0 = torch.tensor([1., 1., 1., 1., 1.], device = 'cuda')
        energies = torch.zeros(len(self.shapes)).cuda()
        arr = arr0/torch.sum(arr0)


        for j in range(len(features_inside)):

            energies[j] = torch.mean(torch.exp(-(features_inside[j] - features_outside[j])**2))
            
        fin = torch.sum(energies * arr0)

        return  fin
    

        
    def forward(self,
                img,
                points_start):
        
        img_anchor = img.astype(np.float32).copy()


        tensor_anchor = torch.unsqueeze(torch.tensor(np.transpose(img_anchor, (-1,0,1)), device = 'cuda'), 0)
        input_tensor = tensor_anchor

        x = self.model(preprocess(input_tensor))
        
    
      
        if img.dtype != 'float64':

            raise Exception("Image must be normalized between 0 and 1")
        

        stop = False
        tots = np.zeros((self.n_epochs))
        scores = np.zeros((self.n_epochs, len(self.shapes)))
        shapes = np.zeros((self.n_epochs, self.nb_points,2))
        
        shape_tmp = img.shape[:-1]
        decalage = np.roll((self.dim - np.array(shape_tmp))//2, 1)
        points_start = (points_start + decalage) / self.dim
        tensor = torch.tensor(np.transpose(img.astype(np.float32),(-1,0,1)).copy(), device = 'cuda')
        input_tensor = torch.unsqueeze(tensor, 0)

        x = self.model(preprocess(input_tensor))
        points_start = self.interpolate(points_start,
                                   self.nb_points)
        shape_init = np.roll(points_start,
                        axis = 1,
                        shift = 1)


        kernel = torch.tensor(self.return_function_exp(1.0, 3), dtype=float)

            
        shape = torch.from_numpy(shape_init).cuda()
        shape.requires_grad = True

        for i in range(self.n_epochs):
                
                tot = self.forward_on_epoch(shape, i = i)
                tots[i] = tot
                shapes[i] = shape.cpu().detach().numpy()

                tot.backward(inputs = shape)

                with torch.no_grad():    

                    norm_grad = torch.unsqueeze(torch.norm(shape.grad, dim = 1),-1)
                    gradient_direction = shape.grad
                    gradient_direction = torch.clip(gradient_direction, - self.clip, self.clip)
                    gradient_direction = self.convolve(gradient_direction,self.kernel).T
                    shape = shape -  self.learning_rate * (self.ed**i) * gradient_direction
                    new_shape = self.interpolate(shape.cpu().detach().numpy(), n = self.nb_points)

                shape = torch.from_numpy(new_shape).cuda()
                shape.grad = None
                shape.requires_grad = True

        shapes = np.roll(shapes,1,-1)
        dec =(self.dim - np.roll(np.array(shape_tmp),1))//2
        shapes = (shapes * self.dim).astype(np.int32)
        shapes = shapes - dec


        return shapes, tots