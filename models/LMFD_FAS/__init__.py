from logging import Logger
from typing import Any, Dict, List

import albumentations
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model.mfad import FAD_HAM_Net


PRE_MEAN = [0.485, 0.456, 0.406]
PRE_STD = [0.229, 0.224, 0.225]


def get_model(config: Dict[str, Any], log: Logger, **kwargs) -> Module:
    log.debug("Initalising LMFD_FAS Model")
    log.debug(f"Provided config: {config}")
    net: Module = FAD_HAM_Net(pretrain=False, variant="resnet50")
    log.debug("Initialised LMFD_FAS Model")

    if kwargs.get("path", ""):
        net.load_state_dict(torch.load(kwargs["path"], weights_only=True))
        log.debug("Loaded weights of LMFD_FAS Model")

    return net


def get_scores(
    data_loader: DataLoader, model: Module, log: Logger, position: int = 0
) -> Dict[str, List[float]]:
    result: Dict[str, List[float]] = {"attack": [], "real": []}

    model.eval()
    model.cuda()
    for x, y in tqdm(data_loader, position=position):
        x = x.cuda()
        y = y.argmax(dim=1).numpy().tolist()

        outputs, _ = model(x)
        pred = torch.sigmoid(outputs).detach().cpu().numpy()

        for prob, lbl in zip(pred, y):
            if lbl:
                result["attack"].append(prob.item())
            else:
                result["real"].append(prob.item())

    return result


def transform_image(fname: str) -> torch.Tensor:
    transform = albumentations.Compose(
        [
            albumentations.SmallestMaxSize(max_size=256),
            albumentations.CenterCrop(height=224, width=224),
            albumentations.Normalize(PRE_MEAN, PRE_STD),
            ToTensorV2(),
        ]
    )

    img = Image.open(fname)
    img = transform(image=np.array(img))["image"]

    return img


__all__ = ["get_model", "get_scores", "transform_image"]
