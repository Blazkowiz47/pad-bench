from enum import Enum
from typing import List

from .metrics import calculate_eer

image_extensions: List[str] = [".jpg", ".png", ".jpeg"]
video_extensions: List[str] = [".mov", ".mp4"]


class SOTA(Enum):
    JPD_FAS = "JPD_FAS"
    FLIP_FAS = "FLIP_FAS"
    LMFD_FAS = "LMFD_FAS"
    DGUA_FAS = "DGUA_FAS"
    CF_FAS = "CF_FAS"
    GACD_FAS = "GACD_FAS"
    IADG_FAS = "IADG_FAS"


__all__ = [
    "image_extensions",
    "video_extensions",
    "calculate_eer",
    "SOTA",
]
