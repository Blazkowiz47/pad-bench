import os
from pathlib import Path
from typing import List, Any, Dict
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import cv2
import numpy as np
import random
import lmdb


image_extensions: List[str] = [".jpg", ".png", ".jpeg"]


class DG_Dataset(Dataset):
    def __init__(
        self,
        data_root=None,
        split="train",
        category=None,
        transform=None,
        img_mode="rgb",
        print_info=False,
        **kwargs,
    ):
        """
        DG_Dataset contain three datasets and preprocess images of three different domains
        Args:
            data_root (str): the path of LMDB_database
            split (str):
                'train': generate the datasets for training
                'val': generate the datasets for validation
                'test': generate the datasets for testing
            catgory (str):
                'pos': generate the datasets of real images
                'neg': generate the datasets of fake images
                '': gnereate the datasets
            transform: the transforms for preprocessing
            img_mode (str):
                'rgb': generate the img in RGB mode
                'rgb_hsv': generate the img in RGB and HSV mode
            print_info (bool):
                'True': print the information of datasets
                'False': do not print the informatino of datasets
        """
        self.data_root = data_root
        self.split = split
        self.category = category
        self.transform = transform
        self.img_mode = img_mode
        self.facedetect = "Face_Detect"

        for k, v in kwargs.items():
            setattr(self, k, v)
        self.rdir = kwargs["rdir"]
        self.attack = kwargs["attack"]

        if self.category == "pos":
            self.items = [f for f in self.loop_splitset(split) if f[1]]
        elif self.category == "neg":
            self.items = [f for f in self.loop_splitset(split) if not f[1]]
        else:
            self.items = self.loop_splitset(split)

        if print_info:
            self._display_infos()

    def loop_splitset(self, ssplit: str) -> List[Any]:
        print(f"Looping through: {self.rdir}")
        data: List[Any] = []
        attack_dir = os.path.join(
            self.rdir,
            "attack",
            self.attack,
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

    def _display_infos(self):
        print(f"=> Dataset {self.__class__.__name__} loaded")
        print(f"=> Split {self.split}")
        print(f"=> category {self.category}")
        print(f"=> Total number of items: {len(self.items)}")
        print(f"=> Image mode: {self.img_mode}")

    def _get_item_index(self, index=0, items=None):
        img_path, label = items[index]
        depth_path = self._convert_to_depth(img_path)

        img = cv2.imread(img_path)

        return img_path, label, img, None, None

    def __getitem_once(self, index, items):
        length = len(items)
        index = index % length

        img_path, label = items[index]

        img = cv2.imread(img_path)
        depth = np.zeros((img.shape[0], img.shape[1]))

        # get hsv or other map
        try:
            if self.img_mode == "rgb":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif self.img_mode == "hsv":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            elif self.img_mode == "ycrcb":
                img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            elif self.img_mode == "rgb_hsv":
                img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            print(img_path)

        # do transform
        if self.transform is not None:
            if self.img_mode == "rgb_hsv":
                results = self.transform(image=img, image0=img_hsv, mask=depth)
                img = results["image"]
                img_hsv = results["image0"]
                depth = results["mask"]
            elif self.img_mode == "rgb":
                results = self.transform(image=img, mask=depth)
                img = results["image"]
                depth = results["mask"]

        if getattr(self, "depth_map", None):
            if getattr(self, "depth_map_size", None):
                size = int(self.depth_map_size)
            else:
                size = 32
            depth = (
                F.interpolate(
                    depth.view(1, 1, depth.shape[0], depth.shape[1]).type(
                        torch.float32
                    ),
                    (size, size),
                    mode="bilinear",
                ).view(size, size)
                / 255
            )

        if self.img_mode == "rgb_hsv":
            img_6ch = torch.cat([img, img_hsv], 0)
            if self.return_path:
                return img_6ch, label, depth, img_path
            else:
                return img_6ch, label, depth

        elif self.img_mode == "rgb":
            if self.return_path:
                return img, label, depth, img_path
            else:
                return img, label, depth

        else:
            print("ERROR: No such img_mode!")
            return

    def __getitem__(self, index):
        if self.split == "train":
            # get data from item_1
            if self.return_path:
                img, label, depth_img, img_dir = self.__getitem_once(index, self.items)
            else:
                img, label, depth_img = self.__getitem_once(index, self.items)
            # get data from item_2

            # need check
            img = torch.unsqueeze(img, dim=0)
            depth_img = torch.unsqueeze(depth_img, dim=0)
            label = torch.from_numpy(np.array([label]))
            if self.return_path:
                img_dir = [img_dir]
        else:
            if self.return_path:
                img, label, depth_img, img_dir = self.__getitem_once(index, self.items)
            else:
                img, label, depth_img = self.__getitem_once(index, self.items)

        if self.return_path:
            return img, label, depth_img, img_dir
        else:
            return img, label, depth_img

    def __len__(self):
        num = len(self.items)
        return num


def set_seed(SEED):
    if SEED:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    from transforms import create_data_transforms
    from omegaconf import OmegaConf

    args = OmegaConf.load("../configs/ICM2O_self.yaml")
    args.dataset.name = "DG_Dataset"
    kwargs = getattr(args.dataset, args.dataset.name)
    set_seed(1234)

    split = "train"
    cagetory = "pos"
    multi = "multi"
    transform = create_data_transforms(args.transform, split, multi)
    train_dataset = DG_Dataset(
        split=split, category=cagetory, transform=transform, **kwargs
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=True
    )
    for i, datas in enumerate(train_dataloader):
        for data in datas:
            print(data.shape) if type(data) is torch.Tensor else len(data)

        # import torchvision as tv
        # tv.utils.save_image(torch.reshape(datas[0],(datas[0].shape[0]*datas[0].shape[1],datas[0].shape[2],datas[0].shape[3],datas[0].shape[4])), 'train_pos_OCMtoI.png', normalize=True)
        # tv.utils.save_image(torch.reshape(datas[1],(datas[1].shape[0]*datas[1].shape[1],datas[1].shape[2],datas[1].shape[3])).unsqueeze(1).type(torch.float32), 'train_pos_OCMtoI_depth.png')
        break

    split = "train"
    cagetory = "neg"
    multi = "multi"
    transform = create_data_transforms(args.transform, split, multi)
    train_dataset = DG_Dataset(
        split=split, category=cagetory, transform=transform, **kwargs
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=True
    )
    for i, datas in enumerate(train_dataloader):
        for data in datas:
            print(data.shape) if type(data) is torch.Tensor else len(data)
        # import torchvision as tv
        # tv.utils.save_image(datas[0], 'train_neg_image_l3_3.png', normalize=True)
        # tv.utils.save_image(datas[1].unsqueeze(1), 'depth.png')
        break

    split = "val"
    cagetory = "pos"
    multi = "multi"
    transform = create_data_transforms(args.transform, split, multi)
    train_dataset = DG_Dataset(
        split=split, category=cagetory, transform=transform, **kwargs
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=True
    )
    for i, datas in enumerate(train_dataloader):
        for data in datas:
            print(data.shape) if type(data) is torch.Tensor else len(data)
        # import torchvision as tv
        # tv.utils.save_image(datas[0], 'test_pos_image_l3_3.png', normalize=True)
        # tv.utils.save_image(datas[1].unsqueeze(1), 'depth.png')
        break

    split = "val"
    cagetory = "neg"
    multi = "multi"
    transform = create_data_transforms(args.transform, split, multi)
    train_dataset = DG_Dataset(
        split=split, category=cagetory, transform=transform, **kwargs
    )
    train_dataloader = DataLoader(
        train_dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=True
    )
    for i, datas in enumerate(train_dataloader):
        for data in datas:
            print(data.shape) if type(data) is torch.Tensor else len(data)
        # import torchvision as tv
        # tv.utils.save_image(datas[0], 'test_neg_image_l3_3.png', normalize=True)
        # tv.utils.save_image(datas[1].unsqueeze(1), 'depth.png')
        break
