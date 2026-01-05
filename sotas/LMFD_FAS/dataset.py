from __future__ import print_function, division
import os
import cv2
import random
from logging import Logger
from pathlib import Path
from typing import List, Callable, Dict, Any

cv2.setNumThreads(0)

import pandas as pd

import torch
from torch.utils.data import Dataset

import albumentations
from albumentations.pytorch import ToTensorV2

image_extensions: List[str] = [".jpg", ".png", ".jpeg"]
PRE__MEAN = [0.485, 0.456, 0.406]
PRE__STD = [0.229, 0.224, 0.225]


def TrainDataAugmentation():
    transforms_train = albumentations.Compose(
        [
            albumentations.SmallestMaxSize(max_size=256),
            albumentations.CenterCrop(height=224, width=224),
            albumentations.HorizontalFlip(),
            albumentations.ShiftScaleRotate(
                shift_limit=0.05, scale_limit=0.05, rotate_limit=0.1, p=0.5
            ),
            albumentations.ColorJitter(
                brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15
            ),
            albumentations.HueSaturationValue(
                hue_shift_limit=15, sat_shift_limit=15, val_shift_limit=15
            ),
            albumentations.Normalize(PRE__MEAN, PRE__STD),
            ToTensorV2(),
        ]
    )
    return transforms_train


def TestDataAugmentation():
    transform_val = albumentations.Compose(
        [
            albumentations.SmallestMaxSize(max_size=256),
            albumentations.CenterCrop(height=224, width=224),
            albumentations.Normalize(PRE__MEAN, PRE__STD),
            ToTensorV2(),
        ]
    )
    return transform_val


class FacePAD_Train(Dataset):
    def __init__(self, image_info_lists, map_size=14):
        self.image_info_lists = pd.read_csv(image_info_lists)
        self.composed_transforms = TrainDataAugmentation()
        self.map_size = map_size

    def __len__(self):
        return len(self.image_info_lists)

    def __getitem__(self, index):
        image_path = self.image_info_lists.iloc[index, 0]
        label_str = self.image_info_lists.iloc[index, 1]

        image_x = cv2.imread(image_path)
        image_x = cv2.cvtColor(image_x, cv2.COLOR_BGR2RGB)

        if label_str == "attack":
            map_x = torch.zeros((self.map_size, self.map_size))
            spoofing_label = 0
        else:
            map_x = torch.ones((self.map_size, self.map_size))
            spoofing_label = 1

        image_x = self.composed_transforms(image=image_x)["image"]

        return (image_x, int(spoofing_label), map_x)


class FacePAD_Val(Dataset):
    def __init__(self, image_info_lists, map_size=14):
        self.image_info_lists = pd.read_csv(image_info_lists)
        self.composed_transforms = TestDataAugmentation()
        self.map_size = map_size

    def __len__(self):
        return len(self.image_info_lists)

    def __getitem__(self, index):
        image_path = self.image_info_lists.iloc[index, 0]
        label_str = self.image_info_lists.iloc[index, 1]

        # obtain video name for computing a mean decision score of all frames of one video
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        video_id = image_name.rsplit("_", 1)[0]

        image_x = cv2.imread(image_path)
        image_x = cv2.cvtColor(image_x, cv2.COLOR_BGR2RGB)

        if label_str == "attack":
            map_x = torch.zeros((self.map_size, self.map_size))
            spoofing_label = 0
        else:
            map_x = torch.ones((self.map_size, self.map_size))
            spoofing_label = 1

        image_x = self.composed_transforms(image=image_x)["image"]

        return (image_x, int(spoofing_label), map_x, video_id)


class StandardWrapper:
    def __init__(
        self,
        config: Dict[str, Any],
        log: Logger,
        map_size: int = 14,
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
        self.map_size = map_size
        self.classes = ["attack", "real"]
        self.num_classes = len(self.classes)
        self.attack_type = kwargs.get("attack", "*")
        self.facedetect = config.get("facedetect", None)
        self.batch_size = config.get("batch_size", 1)
        self.num_workers = config.get("num_workers", 1)
        self.transform = kwargs.get("transform", None)
        self.include_path = kwargs.get("include_path", True)
        self.log.debug(f"Attack: {self.attack_type}")
        print(f"Attack: {self.attack_type}")
        self.train_transform = albumentations.Compose(
            [
                albumentations.SmallestMaxSize(max_size=256),
                albumentations.CenterCrop(height=224, width=224),
                albumentations.HorizontalFlip(),
                albumentations.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=0.1, p=0.5
                ),
                albumentations.ColorJitter(
                    brightness=0.15, contrast=0.15, saturation=0.15, hue=0.15
                ),
                albumentations.HueSaturationValue(
                    hue_shift_limit=15, sat_shift_limit=15, val_shift_limit=15
                ),
                albumentations.Normalize(PRE__MEAN, PRE__STD),
                ToTensorV2(),
            ]
        )
        self.test_transform = albumentations.Compose(
            [
                albumentations.SmallestMaxSize(max_size=256),
                albumentations.CenterCrop(height=224, width=224),
                albumentations.Normalize(PRE__MEAN, PRE__STD),
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
        print(self.facedetect)
        print(attack_dir)
        print(os.path.join(attack_dir, self.facedetect))
        print(os.path.isdir(os.path.join(attack_dir, self.facedetect)))
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

        print(f"Loaded attack files: {len(attackfiles)}")
        print(f"From: {attack_dir}")

        realfiles = [
            str(file)
            for file in Path(real_dir).glob("*")
            if file.suffix.lower() in image_extensions
        ]

        print(f"Loaded real files: {len(realfiles)}")
        print(f"From: {real_dir}")

        balance = min(len(attackfiles), len(realfiles))

        for point in random.sample(attackfiles, k=balance):
            data.append((point, 0))

        for point in random.sample(realfiles, k=balance):
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
            self.map_size,
        )


class DatasetGenerator(Dataset):
    def __init__(self, data: List[Any], transform: Callable, map_size: int) -> None:
        self.data = data
        self.transform = transform
        self.map_size = map_size

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        img_path, label = self.data[index]
        image = cv2.imread(img_path)
        if image is None:
            raise Exception("Error: Image is None.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"]

        if not label:
            map_x = torch.zeros((self.map_size, self.map_size))
        else:
            map_x = torch.ones((self.map_size, self.map_size))
        return image, label, map_x
