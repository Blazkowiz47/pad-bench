from datetime import datetime
from logging import Logger
import os
import random
from abc import abstractmethod
from typing import Any, Callable, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


image_extensions: List[str] = [".jpg", ".png", ".jpeg"]
video_extensions: List[str] = [".mov", ".mp4"]


def set_seeds(log: Logger, seed: int = 2024):
    """
    Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 2024.
    """
    log.debug(f"Setting seed to: {seed}")
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_run_name(model: str, dataset: str) -> str:
    return f"{model}_{dataset}_{str(datetime.now()).replace(' ', '-')}"


def initialise_dirs(model_name: str):
    """
    Initialises all the required directories.
    """
    os.makedirs(model_name, exist_ok=True)


class DatasetGenerator(Dataset):
    def __init__(self, data: List[Any], transform: Callable) -> None:
        self.data = data
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Tuple[Any]:
        datapoint = self.data[index]
        return self.transform(datapoint)


class Wrapper:
    def __init__(
        self,
        log: Logger,
        rdir: str = "",
        batch_size: int = 1,
        num_workers: int = 1,
        num_classes: Optional[int] = None,
    ) -> None:
        self.batch_size = batch_size
        self.log = log
        self.num_workers = num_workers
        self.num_classes = None
        self.rdir = rdir
        self.name = ""

    @abstractmethod
    def loop_splitset(self, ssplit: str, to_augment: bool) -> List[Any]:
        """
        Loops through the given directory.
        Practically, one should only change this function and get various splits.
        Returns: List of files to load along with its class label.
        """
        raise NotImplementedError("")

    @abstractmethod
    def entire_dataset(self, to_augment: bool) -> List[Any]:
        """
        Loops through the given directory.
        Practically, one should only change this function and get various splits.
        Returns: List of files to load along with its class label.
        """
        raise NotImplementedError("")

    def get_split(
        self,
        split: str,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
    ) -> DataLoader:
        """
        Generates the given split.
        """
        self.log.debug(f"Generating {split} split for {self.name} dataset.")
        batch_size = batch_size or self.batch_size
        data = self.loop_splitset(split, split == "train")
        self.log.debug(f"Total files: {len(data)}")
        return DataLoader(
            DatasetGenerator(data, self._transform),
            num_workers=num_workers or self.num_workers,
            batch_size=batch_size or self.batch_size,
        )

    @abstractmethod
    def get(
        self,
        augment: bool = False,
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None,
    ) -> DataLoader:
        """
        Generates the given split.
        """
        self.log.debug(f"Getting {self.name} dataset.")
        batch_size = batch_size or self.batch_size
        data = self.entire_dataset(augment)
        self.log.debug(f"Total files: {len(data)}")
        return DataLoader(
            DatasetGenerator(data, self._transform),
            num_workers=num_workers or self.num_workers,
            batch_size=batch_size or self.batch_size,
        )

    @abstractmethod
    def augment(self, image):
        """
        Augments the given image.
        """
        raise NotImplementedError()

    @abstractmethod
    def _transform(self, datapoint: Iterable[Any]) -> Tuple:
        """
        Transforms the given datapoint.
        """
        raise NotImplementedError()
