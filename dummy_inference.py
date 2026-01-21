# Fake -> 0
# Real -> 1
import argparse
import json
from typing import Any, List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageFile
from torch.nn import DataParallel, Module
from torch.utils.data import DataLoader, Dataset
from torch.multiprocessing import set_start_method
from torchvision import transforms as T
from tqdm import tqdm

try:
    set_start_method("spawn", force=True)
except RuntimeError:
    pass

ImageFile.LOAD_TRUNCATED_IMAGES = True

BATCH_SIZE = 128
PRE_STD = [0.229, 0.224, 0.225]
PRE_MEAN = [0.485, 0.456, 0.406]


def get_model(**kwargs) -> Module:
    """
    Load the model from checkpoint
    expected kwargs: path
    """


class DatasetGenerator(Dataset):
    def __init__(self, data: List[Any]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def transform_image(self, fname: str) -> Any:
        input_shape = (224, 224)
        transform = # Mention the transformation steps here
        img = Image.open(fname)
        img = transform(img)
        return img, fname

    def __getitem__(self, index) -> Any:
        datapoint = self.data[index]
        return self.transform_image(datapoint)


def get_features(model, input):
    """
    Extract features from the model
    """


def collate_fn(batch):
    x, f = [], []
    for item in batch:
        if item[0] is not None:
            x.append(item[0])
        else:
            x.append(torch.zeros(3, 224, 224))

        f.append(item[1])
    return torch.stack(x, dim=0), f


def get_scores(files: List[str], model: Module, position=0) -> List[float]:
    """
    Get the scores for the input files
    """
    result: List[float] = []
    model.eval()
    model.cuda()
    wrapper = DatasetGenerator(files)
    for x, fnames in tqdm(
        DataLoader(
            wrapper,
            batch_size=BATCH_SIZE,
            num_workers=4,
        ),
        position=position,
    ):
        x = x.cuda()
        """
        Perform forward pass and get predictions for real class
        """

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
