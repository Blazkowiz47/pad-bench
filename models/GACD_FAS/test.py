import torch
from logging import getLogger
from .models import get_model, get_scores

def driver() -> None: 
    log = getLogger()
    model = get_model({},log, path='./pretrained_models/GACD_FAS/resnet18_pICM2O_best.pth')
    model.cuda().eval()
    y = model(torch.randn(1, 3,224,224).cuda())
    print(y.shape)

if __name__ == '__main__':
    driver()
