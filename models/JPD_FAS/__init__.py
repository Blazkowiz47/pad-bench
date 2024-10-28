from logging import Logger
from typing import Any, Dict

from torch.nn import Module

from .nets import load_pretrain, get_model as models


def get_model(config: Dict[str, Any], log: Logger, **kwargs) -> Module:
    log.debug("Initalising JPD_FAS Model")
    log.debug(f"Provided config: {config}")
    net = models(config.get("arch", "resnet50"))

    log.debug("Initialised JPD_FAS Model")

    if not isinstance(net, Module):
        raise ValueError("Cannot load JPD_FAS model")

    if kwargs.get("path", ""):
        net = load_pretrain(kwargs["path"], net, log)
        log.debug("Loaded weights of JPD_FAS Model")

    return net


__all__ = ["get_model"]
