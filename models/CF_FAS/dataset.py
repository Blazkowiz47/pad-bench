import os
from logging import Logger
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import albumentations
import cv2
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

image_extensions: List[str] = [".jpg", ".png", ".jpeg"]
PRE_STD = [0.229, 0.224, 0.225]
PRE_MEAN = [0.485, 0.456, 0.406]


def ApplyWeightedRandomSampler(dataset_csv):
    dataframe = pd.read_csv(dataset_csv)  # head: image_path, label
    class_counts = dataframe.label.value_counts()

    sample_weights = [1 / class_counts[i] for i in dataframe.label.values]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(dataframe), replacement=True
    )
    return sampler


class TrainDataset(Dataset):
    def __init__(self, csv_file, input_shape=(224, 224)):
        # self.image_dir = image_dir
        self.dataframe = pd.read_csv(csv_file)
        self.composed_transformations = albumentations.Compose(
            [
                albumentations.Resize(height=256, width=256),
                albumentations.RandomCrop(height=input_shape[0], width=input_shape[0]),
                albumentations.HorizontalFlip(),
                albumentations.RandomGamma(gamma_limit=(80, 180)),  # 0.5, 1.5
                albumentations.RGBShift(
                    r_shift_limit=20, g_shift_limit=20, b_shift_limit=20
                ),
                albumentations.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                ),
                albumentations.Normalize(PRE_MEAN, PRE_STD, always_apply=True),
                ToTensorV2(),
            ]
        )

    def __len__(self):
        return len(self.dataframe)

    def get_labels(self):
        return self.dataframe.iloc[:, 1]

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label_str = self.dataframe.iloc[idx, 1]

        image = cv2.imread(img_path)
        if image is None:
            raise Exception("Error: Image is None.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = 1 if label_str == "bonafide" else 0
        map_x = torch.ones((14, 14)) if label == 1 else torch.zeros((14, 14))

        image = self.composed_transformations(image=image)["image"]

        return {
            "images": image,
            "labels": torch.tensor(label, dtype=torch.float),
            "map": map_x,
        }


class TestDataset(Dataset):
    def __init__(self, csv_file, input_shape=(224, 224)):
        self.dataframe = pd.read_csv(csv_file)
        self.composed_transformations = albumentations.Compose(
            [
                albumentations.Resize(height=input_shape[0], width=input_shape[1]),
                albumentations.Normalize(PRE_MEAN, PRE_STD, always_apply=True),
                ToTensorV2(),
            ]
        )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label_str = self.dataframe.iloc[idx, 1]

        image = cv2.imread(img_path)
        if image is None:
            raise Exception("Error: Image is None.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = 1 if label_str == "bonafide" else 0

        image = self.composed_transformations(image=image)["image"]

        return {
            "images": image,
            "labels": torch.tensor(label, dtype=torch.float),
            "img_path": img_path,
        }


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
        self.train_transform = albumentations.Compose(
            [
                albumentations.Resize(height=256, width=256),
                albumentations.RandomCrop(height=224, width=224),
                albumentations.HorizontalFlip(),
                albumentations.RandomGamma(gamma_limit=(80, 180)),  # 0.5, 1.5
                albumentations.RGBShift(
                    r_shift_limit=20, g_shift_limit=20, b_shift_limit=20
                ),
                albumentations.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                ),
                albumentations.Normalize(PRE_MEAN, PRE_STD, always_apply=True),
                ToTensorV2(),
            ]
        )
        self.test_transform = albumentations.Compose(
            [
                albumentations.Resize(height=224, width=224),
                albumentations.Normalize(PRE_MEAN, PRE_STD, always_apply=True),
                ToTensorV2(),
            ]
        )

    def loop_splitset(self, ssplit: str) -> Tuple[List[Any], Any]:
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
            data.append((point, 0))

        class_counts = {0: len(datapoints)}
        print(f"Loaded attack files: {len(datapoints)}")
        print(f"From: {attack_dir}")

        datapoints = [
            str(file)
            for file in Path(real_dir).glob("*")
            if file.suffix.lower() in image_extensions
        ]

        class_counts = {1: len(datapoints)}
        print(f"Loaded real files: {len(datapoints)}")
        print(f"From: {real_dir}")

        for point in datapoints:
            data.append((point, 1))

        sample_weights = [1 / class_counts[i] for i in class_counts]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(data), replacement=True
        )
        return data, sampler

    def get_split(
        self,
        split: str,
        **kwargs,
    ) -> DataLoader:
        """
        Generates the given split.
        """
        self.log.debug(f"Generating {split} split for {self.name} dataset.")
        if "batch_size" not in kwargs:
            kwargs["batch_size"] = self.batch_size
        if "num_workers" not in kwargs:
            kwargs["num_workers"] = self.num_workers
        data, sampler = self.loop_splitset(split)
        self.log.debug(f"Total files: {len(data)}")

        return DataLoader(
            DatasetGenerator(
                data,
                self.train_transform if split == "train" else self.test_transform,
            ),
            **kwargs,
            sampler=sampler if split == "train" else None,
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

        map_x = torch.ones((14, 14)) if label else torch.zeros((14, 14))
        image = self.transform(image=image)["image"]

        return {
            "images": image,
            "labels": torch.tensor(label, dtype=torch.float),
            "img_path": img_path,
            "map": map_x,
        }
