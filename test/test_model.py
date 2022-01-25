
from async_timeout import sys
import torch
sys.path.insert(0, r"D:\MyProjects\CutPaste")
from model import _CutPasteNetBase, CutPasteNet


def test_model():
    model = CutPasteNet()
    #train 
    a = model.forward(torch.zeros((2, 3, 224, 224)))
    
    #eval
    model.eval()
    b = model.forward(torch.zeros((1, 3, 224, 224)))
    
    assert len(a) == 2
    assert len(b) == 2


def test_gmodel():
    gmodel = _CutPasteNetBase().create_graph_model()
    
    a = gmodel.forward(torch.zeros((2, 3, 224, 224)))
    
    gmodel.eval() 
    b = gmodel.forward(torch.zeros((1, 3, 224, 224)))
    
    assert len(a) == 2
    assert len(b) == 2
    