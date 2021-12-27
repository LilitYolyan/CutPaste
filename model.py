import torch
from torchvision.models import resnet18
import torch.nn as nn

class CutPasteNet(nn.Module):
    def __init__(self, pretrained = True, dims = [512,512,512,512,512,512,512,512,128], num_class = 3):
        super().__init__()
        self.encoder = resnet18(pretrained = pretrained)
        self.encoder.fc = nn.Identity()
        proj_layers = []
        for d in dims[:-1]:
            proj_layers.append(nn.Linear(d,d, bias=False)),
            proj_layers.append((nn.BatchNorm1d(d))),
            proj_layers.append(nn.ReLU(inplace=True))
        embeds = nn.Linear(dims[-2], dims[-1], bias=num_class > 0)
        proj_layers.append(embeds)
        self.head = nn.Sequential(
            *proj_layers
        )
        self.out = nn.Linear(dims[-1], num_class)

    def forward(self, x):
        features = self.encoder(x)
        embeds = self.head(features)
        logits = self.out(embeds)
        return features, logits, embeds

    # def freeze(self, layer_name):
    #     #freeze encoder until layer_name
    #     check = False
    #     for name, param in self.encoder.named_parameters():
    #         if name == layer_name:
    #             check = True 
    #         if not check and param.requires_grad != False:
    #             param.requires_grad = False
    #         else:
    #             param.requires_grad = True

  

    

