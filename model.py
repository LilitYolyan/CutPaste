import torch
import torchvision
from torchvision.models import resnet18
import torch.nn as nn

class Projection(nn.Module):
    def __init__(self, dims = [512,512,512,512,512,512,512,512,128], num_classes = 3):
        super().__init__()
        proj_layers = []
        for d in dims:
            proj_layers.append(nn.Linear(d,d, bias=False)),
            proj_layers.append((nn.BatchNorm1d(d))),
            proj_layers.append(nn.ReLU(inplace=True))
        
        embeds = nn.Linear(dims[-2], dims[-1], bias=num_classes > 0)
        proj_layers.append(embeds)
        self.head = nn.Sequential(
            *proj_layers
        )
        self.out = nn.Linear(dims[-1], num_classes)
            
    def forward(self, x):
        embed = self.head(x)
        logits = self.out(embed)
        return logits, embed




class Encoder(nn.Module):
    def __init__(self, pretrained = True):
        super().__init__()
        self.encoder = resnet18(pretrained = pretrained)
        self.encoder.fc = nn.Identity()

    def forward(self, x):
        x = self.encoder(x)
        return x

    def freeze(self, layer_name):
        #freeze encoder until layer_name
        check = False
        for name, param in self.encoder.named_parameters():
            if name == layer_name:
                check = True 
            if not check and param.requires_grad != False:
                param.requires_grad = False
            else:
                param.requires_grad = True


