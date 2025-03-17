from enum import Enum
from typing import List

# from .common_functions import (
#     DatasetGenerator,
#     Wrapper,
#     get_run_name,
#     initialise_dirs,
#     set_seeds,
# )
from .logger import get_logger
from .metrics import calculate_eer

image_extensions: List[str] = [".jpg", ".png", ".jpeg"]
video_extensions: List[str] = [".mov", ".mp4"]


class SOTA(Enum):
    CF_FAS = "CF_FAS"
    DGUA_FAS = "DGUA_FAS"
    IADG_FAS = "IADG_FAS"
    FLIP_FAS = "FLIP_FAS"
    GACD_FAS = "GACD_FAS"
    JPD_FAS = "JPD_FAS"
    LMFD_FAS = "LMFD_FAS"


__all__ = [
    # "DatasetGenerator",
    # "Wrapper",
    # "get_run_name",
    # "initialise_dirs",
    # "set_seeds",
    "image_extensions",
    "video_extensions",
    "get_logger",
    "calculate_eer",
    "SOTA",
]
