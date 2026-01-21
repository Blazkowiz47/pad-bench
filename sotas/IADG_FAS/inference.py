# Real -> 0 Single class
import argparse
import json
from typing import Any, List, Dict

import albumentations
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
from PIL import Image, ImageFile
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torch.multiprocessing import set_start_method
from tqdm import tqdm

try:
    set_start_method("spawn", force=True)
except RuntimeError:
    pass
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from .models import Framework
except ImportError:
    from models import Framework

BATCH_SIZE = 32
PRE_MEAN = [0.5, 0.5, 0.5]
PRE_STD = [0.5, 0.5, 0.5]


def get_model(**kwargs) -> Module:
    try:
        params: Dict[str, Any] = {
            "in_channels": 3,
            "mid_channels": 384,
            "model_initial": "kaiming",
            "total_dkg_flag": True,
            "style_dim": 384,
            "base_style_num": 128,  # its either 128 or 64
            "concentration_coeff": 0.0078125,
        }

        net: Module = Framework(**params)

        if kwargs.get("path", ""):
            checkpoint = torch.load(kwargs["path"], map_location="cpu")
            state_dict = (
                checkpoint
                if kwargs["path"].endswith("pth")
                else checkpoint["state_dict"]
            )
            net.load_state_dict(state_dict)
        return net

    except RuntimeError:
        params: Dict[str, Any] = {
            "in_channels": 3,
            "mid_channels": 384,
            "model_initial": "kaiming",
            "total_dkg_flag": True,
            "style_dim": 384,
            "base_style_num": 64,  # its either 128 or 64
            "concentration_coeff": 0.0078125,
        }

        net: Module = Framework(**params)

        if kwargs.get("path", ""):
            checkpoint = torch.load(kwargs["path"], map_location="cpu")
            state_dict = (
                checkpoint
                if kwargs["path"].endswith("pth")
                else checkpoint["state_dict"]
            )
            net.load_state_dict(state_dict)
        return net


class DatasetGenerator(Dataset):
    def __init__(self, data: List[Any]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def transform_image(self, fname: str) -> Any:
        transform = albumentations.Compose(
            [
                albumentations.Resize(256, 256),
                albumentations.Normalize(PRE_MEAN, PRE_STD),
                ToTensorV2(),
            ]
        )

        img = Image.open(fname)
        img = transform(image=np.array(img))["image"]
        depth = torch.zeros(img.shape)

        return img, depth, fname

    def __getitem__(self, index) -> Any:
        datapoint = self.data[index]
        return self.transform_image(datapoint)


def get_scores(files: List[str], model: Module, position=0) -> List[float]:
    result: List[float] = []
    model.eval()
    model.cuda()
    wrapper = DatasetGenerator(files)
    for x, depth, fnames in tqdm(
        DataLoader(
            wrapper,
            batch_size=BATCH_SIZE,
            num_workers=4,
        ),
        position=position,
    ):
        x = x.float().cuda()
        depth = depth.float().cuda()

        outputs_catcls, outputs_catdepth, *_ = model(x, None, True, False, False)

        probs = torch.softmax(outputs_catcls["out"], dim=1)[:, 0]
        bs, _, _, _ = outputs_catdepth["out"].shape

        depth_probs = outputs_catdepth["out"].reshape(bs, -1).mean(dim=1)
        pred = (probs + depth_probs).cpu().data.numpy()

        result.extend([(fname, p) for fname, p in zip(fnames, pred.tolist())])

    return result


def driver(ckpt: str, input_json: str, output_json: str, paths: List[str]) -> None:
    if paths:
        files = paths
    elif input_json != "":
        with open(input_json, "r") as fp:
            input = json.load(fp)
        files = input["files"]
    else:
        raise ValueError("Either paths or input_json must be provided")

    model = get_model(path=ckpt)
    result = get_scores(files, model)
    if output_json != "":
        with open(output_json, "w+") as fp:
            json.dump({"result": result}, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ckpt", "--checkpoint", type=str, help="Path to checkpoint", default=""
    )
    parser.add_argument(
        "-i", "--input-json", type=str, help="Path to input json", default=""
    )
    parser.add_argument(
        "-o", "--output-json", type=str, help="Path to output json", default=""
    )
    parser.add_argument(
        "--paths", type=str, nargs="*", help="List of input image paths", default=list()
    )

    args = parser.parse_args()
    checkpoint = args.checkpoint
    input_json = args.input_json
    output_json = args.output_json
    paths = args.paths
    driver(checkpoint, input_json, output_json, paths)
