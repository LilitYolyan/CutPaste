
import cv2
import numpy as np
import torch
import sys
sys.path.insert(0, r"D:\MyProjects\CutPaste")
from grad_cam import GradCam
import requests
from io import BytesIO
from PIL import Image
from icecream import ic

if __name__ == "__main__":
    import json
    from torchvision import transforms
    from torchvision.models import resnet18, ResNet
    from model import _CutPasteNetBase
    
    trans = transforms.Compose([transforms.ToTensor(), transforms.Resize((224, 224))])
    transinv = transforms.ToPILImage()
    
    response1 = requests.get("https://image.freepik.com/free-photo/two-beautiful-puppies-cat-dog_58409-6024.jpg")
    response2 = requests.get("https://www.rover.com/blog/wp-content/uploads/2011/11/german-shepherd-960x540.jpg")
    labels_path = r'D:\MyProjects\CutPaste\test\imagenet.json'
    with open(labels_path) as json_data:
        idx_to_labels = json.load(json_data)
    
    I1 = np.asarray(Image.open(BytesIO(response1.content)))[..., [0, 1, 2]] # RGB
    I2 = np.asarray(Image.open(BytesIO(response2.content)))[..., [0, 1, 2]]
    T1: torch.Tensor = trans(I1).unsqueeze(0)
    T2: torch.Tensor = trans(I2).unsqueeze(0)
    T = torch.cat([T1, T2], dim=0)
    # T = T1
        
    # for resnet test its result
    # for CutPasteNet only test it works or not
    for model in [resnet18(pretrained=True),  _CutPasteNetBase(encoder="resnet18")]:
        ic(str(model.__class__))
        
        names_eval = GradCam.help(model, "eval")  
        ic(names_eval)
        
        name_node = "layer4" if isinstance(model, ResNet) else "encoder.layer4"
        cam = GradCam(model, name_node)
        ic(name_node)
        
        
        fmaps_without_unsample = cam.forward(T, None, with_upsample=False)
        ic(fmaps_without_unsample.shape)
        fmaps_with_upsample = cam.forward(T, None, with_upsample=True)
        ic(fmaps_with_upsample.shape)
        assert fmaps_with_upsample.ndim == 4
        assert fmaps_with_upsample.shape[0] == T.shape[0]
        assert fmaps_with_upsample.shape[-2:] == T.shape[-2:]
        
        
        indices_auto = cam.auto_select_indices(model(T))
        labels_auto = [idx_to_labels[str(idx.item())][1] for idx in indices_auto]
        ic(indices_auto)
        ic(labels_auto)
        
        C = cam.visualize(T) # auto, BCHW
        if isinstance(model, ResNet):
            indices_manual1 = [208, 208]
            indices_manual2 = [283, 283]
            label_manual1 = idx_to_labels[str(208)][1]
            label_manual2 = idx_to_labels[str(283)][1]
            ic(indices_manual1)
            ic(label_manual1)
            ic(indices_manual2)
            ic(label_manual2)
        
            C1 = cam.visualize(T, indices_manual1)
            C2 = cam.visualize(T, indices_manual2)

            cv2.imshow("", np.vstack((np.hstack(C), np.hstack(C1),  np.hstack(C2))))
        else:
            cv2.imshow("", np.hstack(C))
        cv2.waitKey()
        print()
