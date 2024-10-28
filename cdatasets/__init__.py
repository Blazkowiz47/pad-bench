from logging import Logger
from typing import Any, Dict
from utils import Wrapper


def get_dataset(dataset: str, config: Dict[str, Any], log: Logger, **kwargs) -> Wrapper:


    if dataset == "oulu_npu":
        from cdatasets.oulu_npu import Oulu_npuWrapper

        return Oulu_npuWrapper(config, log, **kwargs)


    ### Donot remove this line as the build generator uses this as a marker
    ### while adding new dataset.
    raise NotImplementedError(f"Dataset: {dataset} not present")
