from logging import Logger
from pathlib import Path
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from utils import DatasetGenerator, Wrapper, image_extensions


class Oulu_npuWrapper(Wrapper):
    def __init__(
        self,
        config: Dict[str, Any],
        log: Logger, 
        **kwargs,
    ):
        """
        This is a demo wrapper.
        Update the `loop_splitset` method for looping your dataset in a custom manner.
        By default it fetches data stored in following manner relative to `./data/oulu_npu':
            ROOT-DIR
                -> ClassId1
                    -> train
                        -> Image1.jpg
                        -> Image2.png
                        ...
                    -> test
                        -> Image1.jpeg
                        -> Image2.png
                        ...
                    -> validation
                        -> Image1.jpeg
                        -> Image2.png
                        ...
                -> ClassId2
                    -> train
                        -> Image1.jpg
                        -> Image2.png
                        ...
                    -> test
                        -> Image1.jpg
                        -> Image2.png
                        ...
                    -> validation
                        -> Image1.jpeg
                        -> Image2.png
                        ...
                ...
        """

        self.name = "oulu_npu"
        self.log = log
        self.kwargs: Dict[str, Any] = kwargs
        self.rdir = "./data/oulu_npu"
        self.classes = [
            cid
            for cid in os.listdir(self.rdir)
            if os.path.isdir(os.path.join(self.rdir, cid)) and '.' not in cid
        ]
        self.classes.sort()
        self.num_classes = len(self.classes)

        self.batch_size = config["batch_size"]
        self.num_workers = config["num_workers"]

    def loop_splitset(self, ssplit: str) -> List[Any]:
        data: List[Any] = []
        for cid_lbl, cid in enumerate(self.classes):
            datapoints = [
                str(file)
                for file in Path(os.path.join(self.rdir, cid, ssplit)).glob("*")
                if file.suffix.lower() in image_extensions
            ]
            for point in datapoints:
                data.append((point, cid_lbl))

        return data

    def get_split(
            self, split:str, batch_size: Optional[int] = None, num_workers: Optional[int] = None
    ) -> DataLoader:
        batch_size = batch_size or self.batch_size
        self.log.debug("Looping through %s split." % split)
        data = self.loop_splitset(split)
        self.log.debug("Data-length for %s split: %d" % (split, len(data)))
        return DataLoader(
            DatasetGenerator(data, self.transform),
            num_workers=num_workers or self.num_workers,
            batch_size=batch_size or self.batch_size,
        )

    def augment(self, image: Any) -> Any:
        return image

    def transform(self, datapoint: Iterable[Any]) -> Tuple:
        fname, lbl = datapoint

        # Initialise label
        label = np.zeros((self.num_classes,))
        label[lbl] = 1

        # Initialise image
        img = Image.open(fname)
        imgarray = np.array(img)
        imgarray = cv2.resize(imgarray, [224, 224])
        imgarray = self.augment(imgarray)

        return torch.tensor(imgarray).float(), torch.tensor(label).float()
