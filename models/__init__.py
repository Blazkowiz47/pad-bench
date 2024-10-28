from logging import Logger
from typing import Any, Dict
from torch.nn import Module


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
