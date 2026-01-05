from logging import Logger
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from util import SOTA


def get_model(model: SOTA, config: Dict[str, Any], log: Logger, **kwargs) -> Module:
    """
    add "path" arguement for loading weights from pretrained network
    """
    if model == SOTA.DGUA_FAS:
        from models.DGUA_FAS import get_model

        return get_model(config, log, **kwargs)

    if model == SOTA.GACD_FAS:
        from models.GACD_FAS import get_model

        return get_model(config, log, **kwargs)

    if model == SOTA.JPD_FAS:
        from models.JPD_FAS import get_model

        return get_model(config, log, **kwargs)

    if model == SOTA.CF_FAS:
        from models.CF_FAS import get_model

        return get_model(config, log, **kwargs)

    if model == SOTA.LMFD_FAS:
        from models.LMFD_FAS import get_model

        return get_model(config, log, **kwargs)

    if model == SOTA.FLIP_FAS:
        from models.FLIP_FAS import get_model

        return get_model(config, log, **kwargs)

    if model == SOTA.IADG_FAS:
        from models.IADG_FAS import get_model

        return get_model(config, log, **kwargs)


def get_score_function(
    model: SOTA,
) -> Callable[[DataLoader, Module, Logger, Optional[int]], Dict[str, List[float]]]:
    """ """
    if model == SOTA.DGUA_FAS:
        from models.DGUA_FAS import get_scores

        return get_scores

    if model == SOTA.GACD_FAS:
        from models.GACD_FAS import get_scores

        return get_scores

    if model == SOTA.JPD_FAS:
        from models.JPD_FAS import get_scores

        return get_scores

    if model == SOTA.CF_FAS:
        from models.CF_FAS import get_scores

        return get_scores

    if model == SOTA.LMFD_FAS:
        from models.LMFD_FAS import get_scores

        return get_scores

    if model == SOTA.FLIP_FAS:
        from models.FLIP_FAS import get_scores

        return get_scores

    if model == SOTA.IADG_FAS:
        from models.IADG_FAS import get_scores

        return get_scores


def get_transform_function(
    model: SOTA,
) -> Callable[[str], torch.Tensor]:
    """ """
    if model == SOTA.DGUA_FAS:
        from models.DGUA_FAS import transform_image

        return transform_image

    if model == SOTA.GACD_FAS:
        from models.GACD_FAS import transform_image

        return transform_image

    if model == SOTA.JPD_FAS:
        from models.JPD_FAS import transform_image

        return transform_image

    if model == SOTA.CF_FAS:
        from models.CF_FAS import transform_image

        return transform_image

    if model == SOTA.LMFD_FAS:
        from models.LMFD_FAS import transform_image

        return transform_image

    if model == SOTA.FLIP_FAS:
        from models.LMFD_FAS import transform_image

        return transform_image

    if model == SOTA.IADG_FAS:
        from models.IADG_FAS import transform_image

        return transform_image


def get_finetune_epoch_step(
    model: SOTA,
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
    if model == SOTA.DGUA_FAS:
        from models.DGUA_FAS import finetune_epoch

        return finetune_epoch

    raise NotImplementedError(f"Not implemented: {model.name}")
