from logging import Logger
from typing import Any, Callable, Dict, List
import torch
from torch.nn import Module
from torch.utils.data import DataLoader


def get_model(model: str, config: Dict[str, Any], log: Logger, **kwargs) -> Module:
    """
    add "path" arguement for loading weights from pretrained network
    """
    if model == "DGUA_FAS":
        from models.DGUA_FAS import get_model

        return get_model(config, log, **kwargs)

    if model == "GACD_FAS":
        from models.GACD_FAS import get_model

        return get_model(config, log, **kwargs)

    if model == "JPD_FAS":
        from models.JPD_FAS import get_model

        return get_model(config, log, **kwargs)

    ### Donot remove this line as the build generator uses this as a marker
    ### while adding new model.
    raise NotImplementedError(f"Model: {model} not present")


def get_score_function(
    model: str,
) -> Callable[[DataLoader, Module], Dict[str, List[float]]]:
    """ """
    if model == "DGUA_FAS":
        from models.DGUA_FAS import get_scores

        return get_scores

    if model == "GACD_FAS":
        from models.GACD_FAS import get_scores

        return get_scores

    if model == "JPD_FAS":
        from models.JPD_FAS import get_scores

        return get_scores

    ### Donot remove this line as the build generator uses this as a marker
    ### while adding new model.
    raise NotImplementedError(f"Get score function in: {model} not present")


def get_transform_function(
    model: str,
) -> Callable[[str], torch.Tensor]:
    """ """
    if model == "DGUA_FAS":
        from models.DGUA_FAS import transform_image

        return transform_image

    if model == "GACD_FAS":
        from models.GACD_FAS import transform_image

        return transform_image

    if model == "JPD_FAS":
        from models.JPD_FAS import transform_image

        return transform_image

    ### Donot remove this line as the build generator uses this as a marker
    ### while adding new model.
    raise NotImplementedError(f"Transform function in: {model} not present")
