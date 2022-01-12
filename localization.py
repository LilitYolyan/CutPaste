import torch
from model import CutPasteNet
import math
from anomaly_detection import AnomalyDetection
from dataset import MVTecAD
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from PIL import Image
import math 
import numpy as np
from scipy import signal


class Localize:
    def __init__(self, model_weights, kernel_dim = (32,32), stride = 4, batch_size = 128, device = 'cuda'):
        self.kernel_dim = kernel_dim
        self.stride = stride
        self.batch_size = batch_size
        self.anomaly = AnomalyDetection(model_weights, batch_size)
        self.device = device
        self.transform =  transforms.Compose([transforms.Resize(256,256), ### ADD to arguments
                                             transforms.ToTensor(),
                                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                             ])

    def extract_image_patches(self,image):
        unfold = torch.nn.Unfold(self.kernel_dim, stride=self.stride)
        image_patches = unfold(image).squeeze(0).reshape(-1, 3, *self.kernel_dim )
        batched_patches = torch.split(image_patches, self.batch_size)
        return batched_patches

    def extract_patch_embeddings(self, image):
        patches = self.extract_image_patches(image)
        patch_embeddings =[]
        with torch.no_grad():
            for patch in patches:
                logits, patch_embed = self.anomaly.cutpaste_model(patch.to(self.device))
                patch_embeddings.append(patch_embed.to('cpu'))
                del logits, patch

        patch_dim = math.sqrt(len(patches)*self.batch_size)
        patch_matrix = torch.cat(patch_embeddings).reshape(int(patch_dim), int(patch_dim), -1)
        return patch_matrix


    def patch_GDE_fit(self, train_images, aligned_obj = False):
        dataset = MVTecAD(train_images, mode='test')
        dataloader = DataLoader(dataset, batch_size=1)
        embeds = []
        for img, _ in tqdm(dataloader):
            patch_matrix = self.extract_patch_embeddings(img)
            # TODO
            if aligned_obj:
                pass
            else:
                w,h,c = patch_matrix.shape
                flat = patch_matrix.reshape(w*h, c)
                embeds.append(flat)

        GDE_model = self.anomaly.GDE_fit(torch.cat(embeds))
        return GDE_model

    def patch_heatmap(self,path_to_trian, test_image_pil):
        GDE_model = self.patch_GDE_fit(path_to_trian)

        image = Image.open(test_image_pil)
        test_image = self.transform(image)
        patch_matrix = self.extract_patch_embeddings(test_image)
        w, h, c = patch_matrix.shape
        flat = patch_matrix.reshape(w * h, c)
        score = self.anomaly.GDE_scores(flat, GDE_model)
        score_matrix = score.reshape(1, 1, 57,57) ####### ADD to arguments
        return score_matrix


class Gaussian_smoothing:
    """
        The class does receptive field upsampling via Gaussian smoothing  which
        essentially applies the transposed convolution with the stride of 4, the same stride 
        that is used for dense feature extraction,using a single convolution kernel of size 32×32 
        whose weights are determined by a Gaussian distribution.
        Gaussian kernel generation function is taken from https://github.com/liznerski/fcdd.
    """ 
    def __init__(self, kernel_size=32, stride=4, std=None, device=None):
        self.kernel_size = kernel_size
        self.stride = stride
        self.std = self.kernel_size_to_std() if not std else std
        if device:
            self.device = device 
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    
    def kernel_size_to_std(self):
        """ Returns a standard deviation value for a Gaussian kernel based on its size """
        return np.log10(0.45*self.kernel_size + 1) + 0.25 if self.kernel_size < 32 else 10

    def gkern(self):
        """Returns a 2D Gaussian kernel array with given kernel size k and self.std std """
    
        if self.kernel_size % 2 == 0:
            # if kernel size is even, signal.gaussian returns center values sampled from gaussian at x=-1 and x=1
            # which is much less than 1.0 (depending on std). Instead, sample with kernel size k-1 and duplicate center
            # value, which is 1.0. Then divide whole signal by 2, because the duplicate results in a too high signal.
            gkern1d = signal.gaussian(self.kernel_size - 1, std=self.std).reshape(self.kernel_size - 1, 1)
            gkern1d = np.insert(gkern1d, (self.kernel_size - 1) // 2, gkern1d[(self.kernel_size - 1) // 2]) / 2
        else:
            gkern1d = signal.gaussian(self.kernel_size, std=self.std).reshape(self.kernel_size, 1)
        gkern2d = np.outer(gkern1d, gkern1d)
        return gkern2d
    
    def upsample(self, X):
        tconv = torch.nn.ConvTranspose2d(1,1, kernel_size=self.kernel_size, stride=self.stride)
        tconv.weight.data = torch.from_numpy(self.gkern()).unsqueeze(0).unsqueeze(0).float()
        tconv.to(self.device)
        X = X.to(self.device)
        out = tconv(X)
        return out


