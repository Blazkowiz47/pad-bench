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
        self.rdir = kwargs.get(
            "rdir", "/cluster/nbl-users/Shreyas-Sushrut-Raghu/3D_PAD_Datasets/iPhone11/"
        )
        self.classes = ["attack", "real"]
        self.num_classes = len(self.classes)
        self.attack_type = kwargs.get("attack", "*")
        self.facedetect = config.get("facedetect", None)
        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        self.transform = kwargs.get("transform", None)
        self.include_path = kwargs.get("include_path", False)
        self.log.debug(f"Attack: {self.attack_type}")

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
        if self.facedetect:
            if os.path.isdir(os.path.join(attack_dir, self.facedetect)):
                attack_dir = os.path.join(attack_dir, self.facedetect)
            if os.path.isdir(os.path.join(real_dir, self.facedetect)):
                real_dir = os.path.join(real_dir, self.facedetect)

        datapoints = [
            str(file)
            for file in Path(attack_dir).glob("*")
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

        # Initialise image
        if self.transform:
            imgarray = self.transform(fname)
        else:
            img = Image.open(fname).resize((224, 224))
            imgarray = np.array(img)

        if to_augment:
            imgarray = self.augment(imgarray)

        # Initialise label
        label = np.zeros((self.num_classes,))
        label[lbl] = 1

        if not isinstance(imgarray, torch.Tensor):
            imgarray = torch.tensor(imgarray)

        if self.include_path:
            return imgarray.float(), torch.tensor(label).float(), fname
        else:
            return imgarray.float(), torch.tensor(label).float()
