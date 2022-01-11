import torch
from model import CutPaste


class Localize:
    def __init__(kernel_dim = (32,32), stride = 4, batch_size):
        self.kernel_dim = kernel_dim
        self.stride = stride
        self.batch_size = batch_size
    
    def extract_image_patches(self,image):
        unfold = torch.nn.Unfold((self.kernel_dim), stride=self.tride)
        image_patches = unfold(image).squeeze(0).reshape(-1, self.kernel_dim, 3)
        batched_patches = torch.split(image_patches, self.batch_size)
        return batched_patches

    def extract_patch_embeddings(self, image):
        patches = self.extract_patches(image)
        patch_embeddings =[]
        for patch in patches:
            patch_embed = CutPaste(patch)[-1]
            patch_embeddings.append(patch_embed)
        patch_matrix = torch.tensor(patch_embeddings).reshape((57,57,-1))
        return patch_matrix
            
