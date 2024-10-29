from logging import Logger
from typing import Any, Dict, List

from PIL import Image
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.utils.data import DataLoader
import torchvision.transforms as T

from .resnet import Resnet18


def get_model(config: Dict[str, Any], log: Logger, **kwargs) -> Module:
    log.debug("Initalising GACD_FAS Model")
    log.debug(f"Provided config: {config}")
    net = Resnet18()

    log.debug("Initialised GACD_FAS Model")

    if not isinstance(net, Module):
        raise ValueError("Cannot load GACD_FAS model")

    if kwargs.get("path", ""):
        net_ = torch.load(kwargs["path"])
        net.load_state_dict(net_["state_dict"])
        log.debug("Loaded weights of GACD_FAS Model")

    return net


def get_scores(data_loader: DataLoader, model: Module) -> Dict[str, List[float]]:
    result: Dict[str, List[float]] = {}

    model.eval()
    model.cuda()
    for x, y in data_loader:
        x = x.cuda()
        y = y.numpy().tolist()
        cls_out = model(x)
        pred = F.softmax(cls_out, dim=1).detach().cpu().numpy()
        for prob, lbl in zip(pred, y):
            if lbl:
                result["attack"].append(prob.item())
            else:
                result["real"].append(prob.item())

    return result


def transform_image(fname: str) -> torch.Tensor:
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(fname)
    return transform(img)


__all__ = ["get_model", "get_scores", "transform_image"]
