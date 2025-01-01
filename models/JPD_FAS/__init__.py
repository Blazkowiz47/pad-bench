from logging import Logger
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

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


def get_scores(
    data_loader: DataLoader, model: Module, log: Logger, position: Optional[int] = 0
) -> Dict[str, List[float]]:
    result: Dict[str, List[float]] = {"attack": [], "real": []}

    model.eval()
    model.cuda()
    for x, y in tqdm(data_loader, position=position):
        x = x.cuda()
        y = y.argmax(dim=1).numpy().tolist()
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
    transform = T.Compose(
        [
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),  # RGB [0,255] input, RGB normalize output
        ]
    )

    img = Image.open(fname)

    return transform(img)


__all__ = ["get_model", "get_scores", "transform_image"]
