import os
from pathlib import Path
import numpy as np
import cv2
import random
from torchvision import transforms as T
from torch.utils.data import Dataset, DataLoader
from utils.utils import sample_frames
from PIL import Image, ImageFilter
from typing import Any, Callable, Dict, List, Tuple, Union
from logging import Logger

image_extensions: List[str] = [".jpg", ".png", ".jpeg"]
PRE__MEAN = [0.485, 0.456, 0.406]
PRE__STD = [0.229, 0.224, 0.225]


class simCLRGaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


# FASDataset updated for SSL-CLIP
class FASDatasetSSLCLIP(Dataset):
    def __init__(self, data, transforms=None, train=True):
        self.train = train
        self.photo_path = data[0] + data[1]
        self.photo_label = [0 for i in range(len(data[0]))] + [
            1 for i in range(len(data[1]))
        ]

        # MCIO
        u, indices = np.unique(
            np.array(
                [
                    i.replace("frame0.png", "").replace("frame1.png", "")
                    for i in data[0] + data[1]
                ]
            ),
            return_inverse=True,
        )

        # # WCS
        # u, indices = np.unique(
        #     np.array([
        #         i.replace('00.jpg', '').replace('01.jpg', '').replace('02.jpg', '').replace('03.jpg', '').replace('04.jpg', '').replace('05.jpg', '').replace('06.jpg', '').replace('07.jpg', '').replace('08.jpg', '').replace('09.jpg', '')
        #         for i in data[0] + data[1]
        #     ]),
        #     return_inverse=True)

        self.photo_belong_to_video_ID = indices

        if transforms is None:
            if not train:
                self.transforms = T.Compose(
                    [
                        T.ToTensor(),
                        T.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
            else:
                # normal image
                self.transforms = T.Compose(
                    [
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        T.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )

                # ssl image views (for ssl clip)
                self.ssl_transforms = T.Compose(
                    [
                        T.RandomResizedCrop(224, scale=(0.08, 1.0)),
                        T.RandomApply(
                            [
                                T.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                            ],
                            p=0.8,
                        ),
                        T.RandomGrayscale(p=0.2),
                        T.RandomApply([simCLRGaussianBlur([0.1, 2.0])], p=0.5),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        T.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )

        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.photo_path)

    def __getitem__(self, item):
        if self.train:
            img_path = self.photo_path[item]
            label = self.photo_label[item]
            img = cv2.imread(img_path)
            img = img.astype(np.float32)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            if np.random.randint(2):
                img[..., 1] *= np.random.uniform(0.8, 1.2)
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
            img = Image.fromarray(img.astype(np.uint8)).resize((224, 224))
            # ssl image views (for ssl clip)
            ssl_img_view_1 = self.ssl_transforms(img)
            ssl_img_view_2 = self.ssl_transforms(img)
            # normal image transform
            normal_img = self.transforms(img)
            return normal_img, ssl_img_view_1, ssl_img_view_2, label

        else:
            videoID = self.photo_belong_to_video_ID[item]
            img_path = self.photo_path[item]
            label = self.photo_label[item]
            img = cv2.imread(img_path)
            img = img.astype(np.float32)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img.astype(np.uint8)).resize((224, 224))
            img = self.transforms(img)
            return img, label, videoID, img_path


class FASDataset(Dataset):
    def __init__(self, data, transforms=None, train=True):
        self.train = train
        self.photo_path = data[0] + data[1]
        self.photo_label = [0 for i in range(len(data[0]))] + [
            1 for i in range(len(data[1]))
        ]

        # MCIO
        u, indices = np.unique(
            np.array(
                [
                    i.replace("frame0.png", "").replace("frame1.png", "")
                    for i in data[0] + data[1]
                ]
            ),
            return_inverse=True,
        )

        # # WCS
        # u, indices = np.unique(
        #     np.array([
        #         i.replace('00.jpg', '').replace('01.jpg', '').replace('02.jpg', '').replace('03.jpg', '').replace('04.jpg', '').replace('05.jpg', '').replace('06.jpg', '').replace('07.jpg', '').replace('08.jpg', '').replace('09.jpg', '')
        #         for i in data[0] + data[1]
        #     ]),
        #     return_inverse=True)

        self.photo_belong_to_video_ID = indices

        if transforms is None:
            if not train:
                self.transforms = T.Compose(
                    [
                        T.ToTensor(),
                        T.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
            else:
                self.transforms = T.Compose(
                    [
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        T.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
        else:
            self.transforms = transforms

    def __len__(self):
        return len(self.photo_path)

    def __getitem__(self, item):
        if self.train:
            img_path = self.photo_path[item]
            label = self.photo_label[item]
            img = cv2.imread(img_path)
            img = img.astype(np.float32)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            if np.random.randint(2):
                img[..., 1] *= np.random.uniform(0.8, 1.2)
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
            img = Image.fromarray(img.astype(np.uint8)).resize((224, 224))
            img = self.transforms(img)
            return img, label

        else:
            videoID = self.photo_belong_to_video_ID[item]
            img_path = self.photo_path[item]
            label = self.photo_label[item]
            img = cv2.imread(img_path)
            img = img.astype(np.float32)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img.astype(np.uint8)).resize((224, 224))
            img = self.transforms(img)
            return img, label, videoID, img_path


def get_dataset_one_to_one(
    src1_data,
    src1_train_num_frames,
    src2_data,
    src2_train_num_frames,
    src3_data,
    src3_train_num_frames,
    tgt_data,
    tgt_test_num_frames,
):
    print("Load Source Data")
    print("Source Data: ", src1_data)
    src1_train_data_fake = sample_frames(
        flag=0, num_frames=src1_train_num_frames, dataset_name=src1_data
    )
    src1_train_data_real = sample_frames(
        flag=1, num_frames=src1_train_num_frames, dataset_name=src1_data
    )
    print("Source Data: ", src2_data)
    src2_train_data_fake = sample_frames(
        flag=0, num_frames=src2_train_num_frames, dataset_name=src2_data
    )
    src2_train_data_real = sample_frames(
        flag=1, num_frames=src2_train_num_frames, dataset_name=src2_data
    )
    print("Source Data: ", src3_data)
    src3_train_data_fake = sample_frames(
        flag=2, num_frames=src3_train_num_frames, dataset_name=src3_data
    )
    src3_train_data_real = sample_frames(
        flag=3, num_frames=src3_train_num_frames, dataset_name=src3_data
    )
    print("Load Target Data")
    print("Target Data: ", tgt_data)
    tgt_test_data = sample_frames(
        flag=4, num_frames=tgt_test_num_frames, dataset_name=tgt_data
    )

    batch_size = 3
    src1_train_dataloader_fake = DataLoader(
        FASDataset(src1_train_data_fake, train=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    src1_train_dataloader_real = DataLoader(
        FASDataset(src1_train_data_real, train=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    src2_train_dataloader_fake = DataLoader(
        FASDataset(src2_train_data_fake, train=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    src2_train_dataloader_real = DataLoader(
        FASDataset(src2_train_data_real, train=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    src3_train_dataloader_fake = DataLoader(
        FASDataset(src3_train_data_fake, train=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    src3_train_dataloader_real = DataLoader(
        FASDataset(src3_train_data_real, train=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    batch_size = 10
    tgt_dataloader = DataLoader(
        FASDataset(tgt_test_data, train=False), batch_size=batch_size, shuffle=False
    )

    data_loaders_list = [
        src1_train_dataloader_fake,
        src1_train_dataloader_real,
        src2_train_dataloader_fake,
        src2_train_dataloader_real,
        src3_train_dataloader_fake,
        src3_train_dataloader_real,
        tgt_dataloader,
    ]

    return data_loaders_list


def get_dataset_ssl_clip(
    src1_data,
    src1_train_num_frames,
    src2_data,
    src2_train_num_frames,
    src3_data,
    src3_train_num_frames,
    src4_data,
    src4_train_num_frames,
    src5_data,
    src5_train_num_frames,
    tgt_data,
    tgt_test_num_frames,
):
    print("Load Source Data")
    print("Source Data: ", src1_data)
    src1_train_data_fake = sample_frames(
        flag=0, num_frames=src1_train_num_frames, dataset_name=src1_data
    )
    src1_train_data_real = sample_frames(
        flag=1, num_frames=src1_train_num_frames, dataset_name=src1_data
    )
    print("Source Data: ", src2_data)
    src2_train_data_fake = sample_frames(
        flag=0, num_frames=src2_train_num_frames, dataset_name=src2_data
    )
    src2_train_data_real = sample_frames(
        flag=1, num_frames=src2_train_num_frames, dataset_name=src2_data
    )
    print("Source Data: ", src3_data)
    src3_train_data_fake = sample_frames(
        flag=0, num_frames=src3_train_num_frames, dataset_name=src3_data
    )
    src3_train_data_real = sample_frames(
        flag=1, num_frames=src3_train_num_frames, dataset_name=src3_data
    )
    print("Source Data: ", src4_data)
    src4_train_data_fake = sample_frames(
        flag=0, num_frames=src4_train_num_frames, dataset_name=src4_data
    )
    src4_train_data_real = sample_frames(
        flag=1, num_frames=src4_train_num_frames, dataset_name=src4_data
    )
    print("Source Data: ", src5_data)
    src5_train_data_fake = sample_frames(
        flag=2, num_frames=src5_train_num_frames, dataset_name=src5_data
    )
    src5_train_data_real = sample_frames(
        flag=3, num_frames=src5_train_num_frames, dataset_name=src5_data
    )
    print("Load Target Data")
    print("Target Data: ", tgt_data)
    tgt_test_data = sample_frames(
        flag=4, num_frames=tgt_test_num_frames, dataset_name=tgt_data
    )

    batch_size = 3  # for mcio
    # batch_size = 3 # for wcs
    src1_train_dataloader_fake = DataLoader(
        FASDatasetSSLCLIP(src1_train_data_fake, train=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    src1_train_dataloader_real = DataLoader(
        FASDatasetSSLCLIP(src1_train_data_real, train=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    src2_train_dataloader_fake = DataLoader(
        FASDatasetSSLCLIP(src2_train_data_fake, train=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    src2_train_dataloader_real = DataLoader(
        FASDatasetSSLCLIP(src2_train_data_real, train=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    src3_train_dataloader_fake = DataLoader(
        FASDatasetSSLCLIP(src3_train_data_fake, train=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    src3_train_dataloader_real = DataLoader(
        FASDatasetSSLCLIP(src3_train_data_real, train=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    src4_train_dataloader_fake = DataLoader(
        FASDatasetSSLCLIP(src4_train_data_fake, train=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    src4_train_dataloader_real = DataLoader(
        FASDatasetSSLCLIP(src4_train_data_real, train=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    src5_train_dataloader_fake = DataLoader(
        FASDatasetSSLCLIP(src5_train_data_fake, train=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    src5_train_dataloader_real = DataLoader(
        FASDatasetSSLCLIP(src5_train_data_real, train=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    batch_size = 10
    tgt_dataloader = DataLoader(
        FASDatasetSSLCLIP(tgt_test_data, train=False),
        batch_size=batch_size,
        shuffle=False,
    )

    data_loaders_list = [
        src1_train_dataloader_fake,
        src1_train_dataloader_real,
        src2_train_dataloader_fake,
        src2_train_dataloader_real,
        src3_train_dataloader_fake,
        src3_train_dataloader_real,
        src4_train_dataloader_fake,
        src4_train_dataloader_real,
        src5_train_dataloader_fake,
        src5_train_dataloader_real,
        tgt_dataloader,
    ]

    return data_loaders_list


def get_dataset_one_to_one_ssl_clip(
    src1_data, src1_train_num_frames, tgt_data, tgt_test_num_frames
):
    print("Load Source Data")
    print("Source Data: ", src1_data)
    src1_train_data_fake = sample_frames(
        flag=0, num_frames=src1_train_num_frames, dataset_name=src1_data
    )
    src1_train_data_real = sample_frames(
        flag=1, num_frames=src1_train_num_frames, dataset_name=src1_data
    )

    print("Load Target Data")
    print("Target Data: ", tgt_data)
    tgt_test_data = sample_frames(
        flag=4, num_frames=tgt_test_num_frames, dataset_name=tgt_data
    )

    batch_size = 3
    src1_train_dataloader_fake = DataLoader(
        FASDatasetSSLCLIP(src1_train_data_fake, train=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    src1_train_dataloader_real = DataLoader(
        FASDatasetSSLCLIP(src1_train_data_real, train=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    batch_size = 10
    tgt_dataloader = DataLoader(
        FASDatasetSSLCLIP(tgt_test_data, train=False),
        batch_size=batch_size,
        shuffle=False,
    )

    data_loaders_list = [
        src1_train_dataloader_fake,
        src1_train_dataloader_real,
        tgt_dataloader,
    ]

    return data_loaders_list


def get_dataset(
    src1_data,
    src1_train_num_frames,
    src2_data,
    src2_train_num_frames,
    src3_data,
    src3_train_num_frames,
    src4_data,
    src4_train_num_frames,
    src5_data,
    src5_train_num_frames,
    tgt_data,
    tgt_test_num_frames,
):
    print("Load Source Data")
    print("Source Data: ", src1_data)
    src1_train_data_fake = sample_frames(
        flag=0, num_frames=src1_train_num_frames, dataset_name=src1_data
    )
    src1_train_data_real = sample_frames(
        flag=1, num_frames=src1_train_num_frames, dataset_name=src1_data
    )
    print("Source Data: ", src2_data)
    src2_train_data_fake = sample_frames(
        flag=0, num_frames=src2_train_num_frames, dataset_name=src2_data
    )
    src2_train_data_real = sample_frames(
        flag=1, num_frames=src2_train_num_frames, dataset_name=src2_data
    )
    print("Source Data: ", src3_data)
    src3_train_data_fake = sample_frames(
        flag=0, num_frames=src3_train_num_frames, dataset_name=src3_data
    )
    src3_train_data_real = sample_frames(
        flag=1, num_frames=src3_train_num_frames, dataset_name=src3_data
    )
    print("Source Data: ", src4_data)
    src4_train_data_fake = sample_frames(
        flag=0, num_frames=src4_train_num_frames, dataset_name=src4_data
    )
    src4_train_data_real = sample_frames(
        flag=1, num_frames=src4_train_num_frames, dataset_name=src4_data
    )
    print("Source Data: ", src5_data)
    src5_train_data_fake = sample_frames(
        flag=2, num_frames=src5_train_num_frames, dataset_name=src5_data
    )
    src5_train_data_real = sample_frames(
        flag=3, num_frames=src5_train_num_frames, dataset_name=src5_data
    )
    print("Load Target Data")
    print("Target Data: ", tgt_data)
    tgt_test_data = sample_frames(
        flag=4, num_frames=tgt_test_num_frames, dataset_name=tgt_data
    )

    # batch_size = 8 # for wcs
    batch_size = 3  # for mcio
    src1_train_dataloader_fake = DataLoader(
        FASDataset(src1_train_data_fake, train=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    src1_train_dataloader_real = DataLoader(
        FASDataset(src1_train_data_real, train=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    src2_train_dataloader_fake = DataLoader(
        FASDataset(src2_train_data_fake, train=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    src2_train_dataloader_real = DataLoader(
        FASDataset(src2_train_data_real, train=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    src3_train_dataloader_fake = DataLoader(
        FASDataset(src3_train_data_fake, train=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    src3_train_dataloader_real = DataLoader(
        FASDataset(src3_train_data_real, train=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    src4_train_dataloader_fake = DataLoader(
        FASDataset(src4_train_data_fake, train=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    src4_train_dataloader_real = DataLoader(
        FASDataset(src4_train_data_real, train=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    src5_train_dataloader_fake = DataLoader(
        FASDataset(src5_train_data_fake, train=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    src5_train_dataloader_real = DataLoader(
        FASDataset(src5_train_data_real, train=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )

    batch_size = 10
    tgt_dataloader = DataLoader(
        FASDataset(tgt_test_data, train=False), batch_size=batch_size, shuffle=False
    )

    data_loaders_list = [
        src1_train_dataloader_fake,
        src1_train_dataloader_real,
        src2_train_dataloader_fake,
        src2_train_dataloader_real,
        src3_train_dataloader_fake,
        src3_train_dataloader_real,
        src4_train_dataloader_fake,
        src4_train_dataloader_real,
        src5_train_dataloader_fake,
        src5_train_dataloader_real,
        tgt_dataloader,
    ]

    return data_loaders_list


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
        self.attack_type = kwargs.get("attack", "*")

        self.transform = kwargs.get("transform", None)
        self.include_path = kwargs.get("include_path", False)

        self.classes = ["attack", "real"]
        self.num_classes = len(self.classes)

        self.facedetect = config.get("facedetect", None)
        self.batch_size = config.get("batch_size", 6)
        self.num_workers = config.get("num_workers", 4)

        self.log.debug(f"Attack: {self.attack_type}")
        print(f"Attack: {self.attack_type}")

        self.test_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.train_transform = T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        # ssl image views (for ssl clip)
        self.ssl_transform = T.Compose(
            [
                T.RandomResizedCrop(224, scale=(0.08, 1.0)),
                T.RandomApply(
                    [
                        T.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                    ],
                    p=0.8,
                ),
                T.RandomGrayscale(p=0.2),
                T.RandomApply([simCLRGaussianBlur([0.1, 2.0])], p=0.5),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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
    ) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
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
        if split == "train":
            real = [x for x in data if x[1]]
            attack = [x for x in data if not x[1]]
            return DataLoader(
                DatasetGenerator(
                    real,
                    self.train_transform,
                    self.ssl_transform,
                    self.test_transform,
                    split,
                    self.include_path,
                ),
                batch_size=3,
                shuffle=True,
                drop_last=True,
            ), DataLoader(
                DatasetGenerator(
                    attack,
                    self.train_transform,
                    self.ssl_transform,
                    self.test_transform,
                    split,
                    self.include_path,
                ),
                batch_size=3,
                shuffle=True,
                drop_last=True,
            )

        else:
            return DataLoader(
                DatasetGenerator(
                    data,
                    self.train_transform,
                    self.ssl_transform,
                    self.test_transform,
                    split,
                    True,
                ),
                batch_size=10,
            )


class DatasetGenerator(Dataset):
    def __init__(
        self,
        data: List[Any],
        train_transform: Callable,
        ssl_transform: Callable,
        test_transform: Callable,
        ssplit: str,
        include_path: bool,
    ) -> None:
        self.data = data
        self.train_transform = train_transform
        self.test_transform = test_transform
        self.ssl_transform = ssl_transform
        self.ssplit = ssplit
        self.include_path = include_path

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index):
        img_path, label = self.data[index]

        if self.ssplit == "train":
            img = cv2.imread(img_path)
            img = img.astype(np.float32)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            if np.random.randint(2):
                img[..., 1] *= np.random.uniform(0.8, 1.2)
            img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
            img = Image.fromarray(img.astype(np.uint8)).resize((224, 224))
            img = self.train_transform(img)
            if self.include_path:
                return img, label, img_path

            return img, label

        img = cv2.imread(img_path)
        img = img.astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img.astype(np.uint8)).resize((224, 224))
        img = self.test_transform(img)
        if self.include_path:
            return img, label, img_path
        return img, label
