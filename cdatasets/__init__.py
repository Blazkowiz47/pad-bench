from logging import Logger
from typing import Any, Dict
from util import Wrapper


def get_dataset(dataset: str, config: Dict[str, Any], log: Logger, **kwargs) -> Wrapper:
    """
    Initializes and returns a dataset wrapper based on the specified dataset type.

    Args:
        dataset (str): The name of the dataset to initialize. "standard" supports most of the common datasets.
        config (Dict[str, Any]): Configuration dictionary containing settings for dataset initialization.
        log (Logger): Logger instance for recording dataset operations and processes.
        **kwargs: Additional arguments passed to the dataset wrapper.

    Returns:
        Wrapper: An instance of the dataset wrapper initialized with the given configuration and logging.

    Raises:
        NotImplementedError: If the specified dataset is not supported.

    Notes:
        - This function currently supports only the "standard" dataset, which is wrapped using the
          `FAS_DS_Wrapper` class from `cdatasets.fas_ds_wrapper`.
        - A marker comment is included for extending support to additional datasets.
    """

    if dataset == "standard":
        from cdatasets.fas_ds_wrapper import FAS_DS_Wrapper

        return FAS_DS_Wrapper(config, log, **kwargs)

    ### Donot remove this line as the build generator uses this as a marker
    ### while adding new dataset.
    raise NotImplementedError(f"Dataset: {dataset} not present")
