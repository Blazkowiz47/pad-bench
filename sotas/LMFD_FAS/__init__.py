from logging import Logger
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as A
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
    data_loader: DataLoader, model: Module, log: Logger, position: Optional[int] = 0
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
    transform = A.Compose(
        [
            A.ToTensor(),
            A.Resize((256, 256)),
            A.CenterCrop((224, 224)),
            A.Normalize(mean=PRE_MEAN, std=PRE_STD),
        ]
    )

    img = Image.open(fname)
    img = transform(np.array(img))

    return img


__all__ = ["get_model", "get_scores", "transform_image"]
