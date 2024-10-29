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

        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]
        self.transform = kwargs.get("transform", None)
        self.include_path = kwargs.get("include_path", False)

    def loop_splitset(self, ssplit: str, to_augment: bool) -> List[Any]:
        self.log.debug(f"Looping through: {self.rdir}")
        data: List[Any] = []
        for cid_lbl, cid in enumerate(self.classes):
            if cid == "attack":
                datapoints = []
                for ext in image_extensions:
                    datapoints.extend(
                        glob.glob(
                            os.path.join(
                                self.rdir,
                                cid,
                                self.attack_type,
                                ssplit,
                                "**",
                                f"*{ext}",
                            ),
                            recursive=True,
                        )
                    )
                    datapoints.extend(
                        glob.glob(
                            os.path.join(
                                self.rdir,
                                cid,
                                self.attack_type,
                                ssplit,
                                "**",
                                f"*{ext.upper()}",
                            ),
                            recursive=True,
                        )
                    )
                self.log.debug(f"Loaded attack files: {len(datapoints)}")
                self.log.debug(
                    f"From: {os.path.join(self.rdir, cid, self.attack_type, ssplit)}"
                )
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
