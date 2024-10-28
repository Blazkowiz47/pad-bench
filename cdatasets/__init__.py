from logging import Logger
from typing import Any, Dict
from utils import Wrapper


def get_dataset(dataset: str, config: Dict[str, Any], log: Logger, **kwargs) -> Wrapper:
    if dataset == "standard":
        from cdatasets.fas_ds_wrapper import FAS_DS_Wrapper

        return FAS_DS_Wrapper(config, log, **kwargs)

    ### Donot remove this line as the build generator uses this as a marker
    ### while adding new dataset.
    raise NotImplementedError(f"Dataset: {dataset} not present")
