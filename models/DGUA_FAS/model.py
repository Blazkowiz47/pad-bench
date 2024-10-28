from logging import Logger
from typing import Any, Dict

import cvnets
import torch
from torch.nn import Module

from .option import get_training_arguments


def get_model(config: Dict[str, Any], log: Logger, **kwargs) -> Module:
    log.debug("Initalising DGUA_FAS Model")
    log.debug(f"Provided config: {config}")

    opts = get_training_arguments(
        config_path="./models/DGUA_FAS/configs/mobilevit_s.yaml"
    )
    net = cvnets.get_model(opts)
    log.debug("Initialised DGUA_FAS Model")

    if not isinstance(net, Module):
        raise ValueError("Cannot load DGUA_FAS model")

    if kwargs.get("path", ""):
        net_ = torch.load(kwargs["path"], weights_only=False)
        net.load_state_dict(net_["state_dict"])
        log.debug("Loaded weights of DGUA_FAS Model")

    return net
