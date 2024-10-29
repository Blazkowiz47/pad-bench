from logging import Logger
from pathlib import Path
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from util import DatasetGenerator, Wrapper, image_extensions


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

        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        self.transform = kwargs.get("transform", None)
        self.include_path = kwargs.get("include_path", False)

    def loop_splitset(self, ssplit: str, to_augment: bool) -> List[Any]:
        data: List[Any] = []
        for cid_lbl, cid in enumerate(self.classes):
            if cid == "attack":
                datapoints = [
                    str(file)
                    for file in Path(
                        os.path.join(self.rdir, cid, self.attack_type, ssplit)
                    ).rglob("*")
                    if file.suffix.lower() in image_extensions
                ]
            else:
                datapoints = [
                    str(file)
                    for file in Path(os.path.join(self.rdir, cid, ssplit)).rglob("*")
                    if file.suffix.lower() in image_extensions
                ]
            for point in datapoints:
                data.append((point, cid_lbl, to_augment))

        return data

    def augment(self, image: Any) -> Any:
        return image

    def _transform(self, datapoint: Iterable[Any]) -> Tuple:
        fname, lbl, to_augment = datapoint

        # Initialise image
        if self.transform:
            imgarray = self.transform(fname)
        else:
            img = Image.open(fname)
            imgarray = np.array(img)
            imgarray = cv2.resize(imgarray, [224, 224])

        if to_augment:
            imgarray = self.augment(imgarray)

        # Initialise label
        label = np.zeros((self.num_classes,))
        label[lbl] = 1

        if self.include_path:
            return torch.tensor(imgarray).float(), torch.tensor(label).float(), fname
        else:
            return torch.tensor(imgarray).float(), torch.tensor(label).float()
