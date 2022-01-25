


from typing import Iterable, List, Optional
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models.feature_extraction import get_graph_node_names
import captum.attr



class GradCam(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, name_layer: str) -> None:
        """GradCam

        Args:
            model (torch.nn.Module): input model.
            name_layer (str): node name of layer interested in
        """        
        super().__init__()
        self.model = model
        self.model.eval()
        
        names_mid = name_layer.split(".")
        layer = model
        for name in names_mid:
            layer = layer.__getattr__(name)
        self.layer = layer
        
        self.cam = captum.attr.LayerGradCam(self.model, self.layer)
        return
    
    def forward(self, x: torch.Tensor, indices: Optional[Iterable[int]]=None, with_upsample: bool = False):
        """[summary]

        Args:
            x (torch.Tensor): input images, [B, C, H, W]
            indices (Optional[Iterable[int]], optional): indices of labels. Defaults to None.
            with_upsample (bool, optional): whether upsample featuremaps to image field. Defaults to False.

        Returns:
            featuremaps (torch.Tensor): output featuremaps, 
                [B, 1, H, W] if with_upsample == True
                [B, 1, _, _] if with_upsample == False
        """        
        if indices is None:
            indices = self.auto_select_indices(self.model.forward(x))
        else:
            pass
        x = x.requires_grad_(True)
        featuremaps =  self.cam.attribute(x, indices, relu_attributions=True)
        featuremaps = self.upsample(featuremaps, size_dst=x.shape[-2:]) if with_upsample else featuremaps
        return featuremaps
    
    def upsample(self, x: torch.Tensor, size_dst: Iterable[int], method="bilinear"):
        x = F.interpolate(input=x, size=size_dst, mode=method, align_corners=True)
        return x
    
    @staticmethod
    def auto_select_indices(logits: torch.Tensor, with_softmax: bool = True) -> torch.Tensor:
        """Auto selct indices of categroies with max probability.

        Args:
            logits (torch.Tensor): [B, C, ...]
            with_softmax (bool, optional): use softmax or not. Defaults to True.

        Returns:
            indices (torch.Tensor): [B, ]
        """
        props = F.softmax(logits, dim=1) if with_softmax else logits
        indices = torch.argmax(props, dim=1, keepdim=False)
        return indices
    
    @staticmethod
    def featuremaps_to_heatmaps(x: torch.Tensor) -> np.ndarray:
        """Convert featuremaps to heatmaps in BGR.

        Args:
            x (torch.Tensor): featuremaps of grad cam, [B, 1, H, W]

        Returns:
            heatmaps (np.ndarray): heatmaps, [B, H, W, C] in BGR
        """        
        (B, _, H, W) = x.shape
        featuremaps = x.squeeze(1).detach().cpu().numpy()
        heatmaps = np.zeros((B, H, W, 3), dtype=np.uint8)
        for (i_map, fmap) in enumerate(featuremaps):
            hmap = cv2.normalize(fmap, None, 0, 1, cv2.NORM_MINMAX)
            hmap = cv2.convertScaleAbs(hmap, None, 255, None)
            hmap = cv2.applyColorMap(hmap, cv2.COLORMAP_JET)
            heatmaps[i_map, :, :] = hmap
        return heatmaps
    
    @staticmethod
    def help(model: torch.nn.Module, mode: str = "eval") -> List[str]:
        """Show valid node names of model.

        Args:
            model (torch.nn.Module): [description]
            mode (str): "eval" or "train", Default is "eval"

        Returns:
            names (List[str]): 
                valid train node names, if mode == "train"
                valid eval node names, if mode == "eval"
        """        
        (names_train, names_eval) = get_graph_node_names(model)
        if mode == "eval":
            return names_eval
        elif mode == "train":
            return names_train
        
    
    def visualize(self, x: torch.Tensor, indices: Optional[int]=None) -> np.ndarray:
        """Visualize heatmaps on raw images.  
            
        Args:
            x (torch.Tensor): input images, [B, C, H, W] in RGB
            indices (Optional[Iterable[int]], optional): indices of labels. Defaults to None.
                if indices is None, it will be auto selected.

        Returns:
            images_show (np.ndarray): input images, [B, H, W, 3] in BGR
        """        
        (B, _, H, W) = x.shape
        images_show = np.zeros((B, H, W, 3), dtype=np.uint8)
        images_raw  = x.permute((0, 2, 3, 1))[..., [2, 1, 0]].detach().cpu().numpy() 
        images_raw  = (images_raw * 255).astype(np.uint8)
        heatmaps = GradCam.featuremaps_to_heatmaps(self.forward(x, indices=indices, with_upsample=True))
        images_show = cv2.addWeighted(images_raw, 0.7, heatmaps, 0.3, 0)
        return images_show
    


