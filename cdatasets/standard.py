import glob
import os
from logging import Logger
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image

from util import Wrapper, image_extensions


class StandardWrapper(Wrapper):
    def __init__(
        self,
        config: Dict[str, Any],
        log: Logger,
        **kwargs,
    ):
        """ """

        self.name = "standard"
        self.log = log
        self.kwargs: Dict[str, Any] = kwargs
        self.rdir = kwargs.get("rdir", "/home/ubuntu/datasets/test/")
        self.classes = ["attack", "real"]
        self.num_classes = len(self.classes)
        self.attack_type = kwargs.get("attack", "*")
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        self.transform = kwargs.get("transform", None)
        self.include_path = kwargs.get("include_path", False)
        self.log.debug(f"Attack: {self.attack_type}")

    def entire_dataset(self, to_augment: bool) -> List[Any]:
        self.log.debug(f"Looping through: {self.rdir}")
        data: List[Any] = []
        attack_dir = os.path.join(
            self.rdir,
            "attack",
        )
        real_dir = os.path.join(
            self.rdir,
            "real",
        )

        datapoints = [
            str(file)
            for file in Path(attack_dir).rglob("*")
            if file.suffix.lower() in image_extensions
        ]

        for point in datapoints:
            data.append((point, 1, to_augment))
        self.log.debug(f"Loaded attack files: {len(datapoints)}")
        self.log.debug(f"From: {attack_dir}")

        datapoints = [
            str(file)
            for file in Path(real_dir).rglob("*")
            if file.suffix.lower() in image_extensions
        ]

        self.log.debug(f"Loaded real files: {len(datapoints)}")
        self.log.debug(f"From: {real_dir}")
        for point in datapoints:
            data.append((point, 0, to_augment))
        return data

    def loop_splitset(self, ssplit: str, to_augment: bool) -> List[Any]:
        self.log.debug(f"Looping through: {self.rdir}")
        data: List[Any] = []
        attack_dir = os.path.join(
            self.rdir,
            "attack",
            self.attack_type,
            ssplit,
        )
        real_dir = os.path.join(
            self.rdir,
            "real",
            ssplit,
        )

        datapoints = [
            str(file)
            for file in Path(attack_dir).rglob("*")
            if file.suffix.lower() in image_extensions
        ]

        for point in datapoints:
            data.append((point, 1, to_augment))
        self.log.debug(f"Loaded attack files: {len(datapoints)}")
        self.log.debug(f"From: {attack_dir}")

        datapoints = [
            str(file)
            for file in Path(real_dir).glob("*")
            if file.suffix.lower() in image_extensions
        ]

        self.log.debug(f"Loaded real files: {len(datapoints)}")
        self.log.debug(f"From: {real_dir}")
        for point in datapoints:
            data.append((point, 0, to_augment))
        return data

    def augment(self, image: Any) -> Any:
        return image

    def _transform(self, datapoint: Iterable[Any]) -> Tuple:
        fname, lbl, to_augment = datapoint
        # Initialise label
        label = np.zeros((self.num_classes,))
        label[lbl] = 1

        # Initialise image
        if self.transform:
            imgarray = self.transform(fname)

        else:
            img = Image.open(fname).resize((224, 224))
            imgarray = np.array(img)

        if to_augment:
            imgarray = self.augment(imgarray)

        if isinstance(imgarray, tuple):
            if self.include_path:
                return (*imgarray, torch.tensor(label).float(), fname)
            else:
                return (*imgarray, torch.tensor(label).float())

        if not isinstance(imgarray, torch.Tensor):
            imgarray = torch.tensor(imgarray)

        if self.include_path:
            return imgarray.float(), torch.tensor(label).float(), fname
        else:
            return imgarray.float(), torch.tensor(label).float()
