from enum import Enum

from .common_functions import (
    DatasetGenerator,
    Wrapper,
    get_run_name,
    image_extensions,
    initialise_dirs,
    set_seeds,
    video_extensions,
)
from .logger import get_logger
from .metrics import calculate_eer


class SOTA(Enum):
    CF_FAS = "CF_FAS"
    DGUA_FAS = "DGUA_FAS"
    IADG_FAS = "IADG_FAS"
    FLIP_FAS = "FLIP_FAS"
    GACD_FAS = "GACD_FAS"
    JPD_FAS = "JPD_FAS"
    LMFD_FAS = "LMFD_FAS"


__all__ = [
    "DatasetGenerator",
    "Wrapper",
    "get_run_name",
    "image_extensions",
    "initialise_dirs",
    "set_seeds",
    "video_extensions",
    "get_logger",
    "calculate_eer",
    "SOTA",
]
