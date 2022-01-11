import torch
from model import CutPasteNet
import math 

class Localize:
    def __init__(self, kernel_dim = (32,32), stride = 4, batch_size = 4):
        self.kernel_dim = kernel_dim
        self.stride = stride
        self.batch_size = batch_size
        self.model = CutPasteNet(pretrained=True)
    
    def extract_image_patches(self,image):
        unfold = torch.nn.Unfold(self.kernel_dim, stride=self.stride)
        image_patches = unfold(image).squeeze(0).reshape(-1, *self.kernel_dim, 3)
        batched_patches = torch.split(image_patches, self.batch_size)
        return batched_patches

    def extract_patch_embeddings(self, image):
        patches = self.extract_image_patches(image)
        patch_embeddings =[]
        for patch in patches:
            print(patch.shape)
            patch_embed = self.model(patch)[-1]
            patch_embeddings.append(patch_embed)
        patch_dim = math.sqrt(len(patches)*self.batch_size)
        patch_matrix = torch.tensor(patch_embeddings).reshape((patch_dim, patch_dim)-1)
        return patch_matrix
            

