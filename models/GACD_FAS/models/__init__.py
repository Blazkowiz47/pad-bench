from logging import Logger
from typing import Any, Dict

import torch
from torch.nn import Module

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


__all__ = ["get_model"]
