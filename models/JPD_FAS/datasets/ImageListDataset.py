import os
import torch
import cv2
import random
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from logging import Logger
from typing import List, Dict, Any, Callable
import albumentations as A
from albumentations.pytorch import ToTensorV2

image_extensions: List[str] = [".jpg", ".png", ".jpeg"]
PRE_STD = [0.229, 0.224, 0.225]
PRE_MEAN = [0.485, 0.456, 0.406]


def is_valid_jpg(jpg_file):
    if not os.path.exists(jpg_file):
        return False
    if jpg_file.split(".")[-1].lower() in ["jpg", "jpeg"]:
        with open(jpg_file, "rb") as f:
            f.seek(-2, 2)
            return f.read() == b"\xff\xd9"
    else:
        return True


class ImageListDataset(Dataset):
    def __init__(
        self,
        basedir,
        data_list,
        is_train,
        transforms1=None,
        transforms2=None,
        return_path=False,
    ):
        self.basedir = basedir
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.return_path = return_path
        self.is_train = is_train
        fp = open(data_list, "rt")
        lines = fp.readlines()
        lines = [line.strip() for line in lines if len(line.strip()) > 0]
        fp.close()

        self.items = []
        for line in lines:
            if is_train:
                items = line.strip().split()
                img_path = items[0]
                label = items[1]
                self.items.append((img_path, label))
            else:
                items = line.strip().split()
                img_path = items[0]
                if len(items) == 1:
                    label = "1"
                else:
                    label = items[1]
                self.items.append((img_path, label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        while True:
            img_path, label = self.items[idx]
            image = cv2.imread(os.path.join(self.basedir, img_path))
            if image is None:
                print("image read error, {}".format(img_path))
                idx = random.randrange(0, len(self.items))
                continue
            break

        if self.transforms1 is not None:
            image = self.transforms1(image)
        if self.transforms2 is not None:
            image = self.transforms2(image=image)
            image = image["image"]

        if self.return_path:
            return image, int(label), img_path
        else:
            return image, int(label)


class StandardWrapper:
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
            "rdir",
            "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/3D_PAD_Datasets/2D_Face_Databases_PAD/iPhone12/",
        )
        self.classes = ["attack", "real"]
        self.num_classes = len(self.classes)
        self.attack_type = kwargs.get("attack", "*")
        self.facedetect = config.get("facedetect", None)
        self.batch_size = config.get("batch_size", 1)
        self.num_workers = config.get("num_workers", 1)
        self.transform = kwargs.get("transform", None)
        self.include_path = kwargs.get("include_path", True)
        self.log.debug(f"Attack: {self.attack_type}")
        self.train_transform = A.Compose(
            [
                A.Resize(height=256, width=256),
                A.RandomCrop(height=224, width=224),
                A.HorizontalFlip(),
                A.RandomGamma(gamma_limit=(80, 180)),  # 0.5, 1.5
                A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20),
                A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                A.Normalize(PRE_MEAN, PRE_STD, always_apply=True),
                ToTensorV2(),
            ]
        )
        self.test_transform = A.Compose(
            [
                A.Resize(height=224, width=224),
                A.Normalize(PRE_MEAN, PRE_STD, always_apply=True),
                ToTensorV2(),
            ]
        )

    def loop_splitset(self, ssplit: str) -> List[Any]:
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

        attackfiles = [
            str(file)
            for file in Path(attack_dir).glob("*")
            if file.suffix.lower() in image_extensions
        ]

        self.log.debug(f"Loaded attack files: {len(attackfiles)}")
        self.log.debug(f"From: {attack_dir}")

        realfiles = [
            str(file)
            for file in Path(real_dir).glob("*")
            if file.suffix.lower() in image_extensions
        ]

        self.log.debug(f"Loaded real files: {len(realfiles)}")
        self.log.debug(f"From: {real_dir}")

        balance = min(len(attackfiles), len(realfiles))

        for point in attackfiles:
            data.append((point, 0))

        for point in realfiles:
            data.append((point, 1))

        return data

    def get_split(
        self,
        split: str,
        **kwargs,
    ) -> Dataset:
        """
        Generates the given split.
        """
        self.log.debug(f"Generating {split} split for {self.name} dataset.")
        if "batch_size" not in kwargs:
            kwargs["batch_size"] = self.batch_size
        if "num_workers" not in kwargs:
            kwargs["num_workers"] = self.num_workers
        data = self.loop_splitset(split)
        self.log.debug(f"Total files: {len(data)}")

        return DatasetGenerator(
            data,
            self.train_transform if split == "train" else self.test_transform,
        )


class DatasetGenerator(Dataset):
    def __init__(self, data: List[Any], transform: Callable) -> None:
        self.data = data
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        img_path, label = self.data[index]
        image = cv2.imread(img_path)
        if image is None:
            raise Exception("Error: Image is None.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"]
        return image, label
