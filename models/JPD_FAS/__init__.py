from logging import Logger
from typing import Any, Dict, List

import albumentations as A
import cv2
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.utils.data import DataLoader

from .cv2_transform import transforms
from .nets import get_model as models
from .nets import load_pretrain


def get_model(config: Dict[str, Any], log: Logger, **kwargs) -> Module:
    log.debug("Initalising JPD_FAS Model")
    log.debug(f"Provided config: {config}")
    net = models(config.get("arch", "resnet50"))
    net._arch = config.get("arch", "resnet50")

    log.debug("Initialised JPD_FAS Model")

    if not isinstance(net, Module):
        raise ValueError("Cannot load JPD_FAS model")

    if kwargs.get("path", ""):
        net = load_pretrain(kwargs["path"], net, log)
        log.debug("Loaded weights of JPD_FAS Model")

    return net


def get_scores(data_loader: DataLoader, model: Module) -> Dict[str, List[float]]:
    result: Dict[str, List[float]] = {}

    model.eval()
    model.cuda()
    for x, y in data_loader:
        x = x.cuda()
        y = y.numpy().tolist()
        if model._arch.startswith("mobilenet") or model._arch.startswith("shufflenet"):
            outputs = model(x)
        else:
            _, outputs = model(x)

        pred = F.softmax(outputs, dim=1)[:, 0].detach().cpu().numpy()
        for prob, lbl in zip(pred, y):
            if lbl:
                result["attack"].append(prob.item())
            else:
                result["real"].append(prob.item())

    return result


def transform_image(fname: str) -> torch.Tensor:
    transform1 = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ColorTrans(mode=0),  # BGR to RGB
    ])
    transform2 = A.Compose([
        A.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ),  # RGB [0,255] input, RGB normalize output
        A.pytorch.ToTensorV2(),
    ])
    img = cv2.imread(fname)
    return transform2(transform1(img))


__all__ = ["get_model", "get_scores", "transform_image"]
