from logging import Logger
from typing import Any, Dict, List

import albumentations
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from torch.nn import DataParallel
from PIL import Image
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from .model import MixStyleResCausalModel

PRE_STD = [0.229, 0.224, 0.225]
PRE_MEAN = [0.485, 0.456, 0.406]


def get_model(config: Dict[str, Any], log: Logger, **kwargs) -> Module:
    log.debug("Initalising CF_FAS Model")
    log.debug(f"Provided config: {config}")
    net: Module = DataParallel(MixStyleResCausalModel())
    log.debug("Initialised CF_FAS Model")

    if kwargs.get("path", ""):
        net.load_state_dict(torch.load(kwargs["path"], weights_only=True))
        log.debug("Loaded weights of CF_FAS Model")

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

        outputs = model(x, cf=None)
        pred = outputs.softmax(dim=1)[:, 1].detach().cpu().numpy()
        for prob, lbl in zip(pred, y):
            if lbl:
                result["attack"].append(prob.item())
            else:
                result["real"].append(prob.item())

    return result


def transform_image(fname: str) -> torch.Tensor:
    input_shape = (224, 224)
    transform = albumentations.Compose(
        [
            albumentations.Resize(height=input_shape[0], width=input_shape[1]),
            albumentations.Normalize(PRE_MEAN, PRE_STD, always_apply=True),
            ToTensorV2(),
        ]
    )

    img = Image.open(fname)
    img = transform(image=np.array(img))["image"]

    return img


__all__ = ["get_model", "get_scores", "transform_image"]
