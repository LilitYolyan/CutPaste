import torch
from model import CutPasteNet
import math
from anomaly_detection import AnomalyDetection
from dataset import MVTecAD
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from PIL import Image

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
        #self.model = CutPasteNet(pretrained=True)
    
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

l = Localize('/home/lilityolyan/stuff/cutpaste/tb_logs/bottle_loc/version_0/checkpoints/weights-bottle.ckpt')
img = torch.rand(1,3,256,256)
#print(l.extract_patch_embeddings(img).shape)
print(l.patch_heatmap('/media/lilityolyan/DATA/damage/mvtec/bottle'))