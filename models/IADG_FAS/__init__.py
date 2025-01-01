from logging import Logger
from typing import Any, Dict, List, Optional, Tuple

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

from .models import Framework
from .dataloaders import transforms as dT

PRE_MEAN = [0.5, 0.5, 0.5]
PRE_STD = [0.5, 0.5, 0.5]


def get_model(config: Dict[str, Any], log: Logger, **kwargs) -> Module:
    log.debug("Initalising IADG_FAS Model")
    params: Dict[str, Any] = {
        "in_channels": 3,
        "mid_channels": 384,
        "model_initial": "kaiming",
        "total_dkg_flag": True,
        "style_dim": 384,
        "base_style_num": 128,
        "concentration_coeff": 0.0078125,
    }
    log.debug(f"Provided config: {config} and loading with: {params}")

    net: Module = Framework(**params)
    log.debug("Initialised IADG_FAS Model")

    if kwargs.get("path", ""):
        checkpoint = torch.load(kwargs["path"], map_location="cpu")
        state_dict = (
            checkpoint if kwargs["path"].endswith("pth") else checkpoint["state_dict"]
        )
        net.load_state_dict(state_dict)
        log.debug("Loaded weights of IADG_FAS Model")

    return net


def get_scores(
    data_loader: DataLoader, model: Module, log: Logger, position: Optional[int] = 0
) -> Dict[str, List[float]]:
    result: Dict[str, List[float]] = {"attack": [], "real": []}

    model.eval()
    model.cuda()
    log.debug("Starting eval Loop")
    for x, depth, y in tqdm(data_loader, position=position):
        x = x.cuda().float()
        depth = depth.cuda().float()
        y = y.argmax(dim=1).numpy().tolist()

        outputs_catcls, outputs_catdepth, *_ = model(x, None, True, False, False)
        log.debug("Forward pass done")

        probs = torch.softmax(outputs_catcls["out"], dim=1)[:, 0]
        bs, _, _, _ = outputs_catdepth["out"].shape

        depth_probs = outputs_catdepth["out"].reshape(bs, -1).mean(dim=1)
        pred = (probs + depth_probs).cpu().data.numpy()

        for prob, lbl in zip(pred, y):
            if lbl:
                result["attack"].append(prob.item())
            else:
                result["real"].append(prob.item())

    return result


def transform_image(fname: str) -> Tuple[torch.Tensor, torch.Tensor]:
    transform = albumentations.Compose(
        [
            albumentations.Resize(256, 256),
            albumentations.Normalize(PRE_MEAN, PRE_STD),
            ToTensorV2(),
        ]
    )

    img = Image.open(fname)
    img = transform(image=np.array(img))["image"]
    depth = torch.zeros(img.shape)

    return img, depth


__all__ = ["get_model", "get_scores", "transform_image"]
