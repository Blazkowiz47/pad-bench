from logging import Logger
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.nn import Module
from torch.optim import Optimizer
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

    if model == "CF_FAS":
        from models.CF_FAS import get_model

        return get_model(config, log, **kwargs)

    if model == "LMFD_FAS":
        from models.LMFD_FAS import get_model

        return get_model(config, log, **kwargs)

    ### Donot remove this line as the build generator uses this as a marker
    ### while adding new model.
    raise NotImplementedError(f"Model: {model} not present")


def get_score_function(
    model: str,
) -> Callable[[DataLoader, Module, Logger, int], Dict[str, List[float]]]:
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

    if model == "CF_FAS":
        from models.CF_FAS import get_scores

        return get_scores

    if model == "LMFD_FAS":
        from models.LMFD_FAS import get_scores

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

    if model == "CF_FAS":
        from models.CF_FAS import transform_image

        return transform_image

    if model == "LMFD_FAS":
        from models.LMFD_FAS import transform_image

        return transform_image

    ### Donot remove this line as the build generator uses this as a marker
    ### while adding new model.
    raise NotImplementedError(f"Transform function in: {model} not present")


def get_finetune_epoch_step(
    model: str,
) -> Callable[
    [
        Module,
        DataLoader,
        Optimizer,
        Module,
        Logger,
        str,
    ],
    Tuple[float, float],
]:
    """ """
    if model == "DGUA_FAS":
        from models.DGUA_FAS import finetune_epoch

        return finetune_epoch

    if model == "GACD_FAS":
        from models.GACD_FAS import transform_image

        return transform_image

    if model == "JPD_FAS":
        from models.JPD_FAS import transform_image

        return transform_image

    if model == "CF_FAS":
        from models.CF_FAS import transform_image

        return transform_image

    if model == "LMFD_FAS":
        from models.LMFD_FAS import transform_image

        return transform_image

    ### Donot remove this line as the build generator uses this as a marker
    ### while adding new model.
    raise NotImplementedError(f"Finetune epoch function in: {model} not present")
