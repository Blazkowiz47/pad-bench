# Real -> 0
# Fake -> 1

import argparse
import traceback
import json
from typing import Any, List, Tuple
from logging import getLogger

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torch.multiprocessing import set_start_method
from torchvision import transforms as T
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from .nets import get_model as models, load_pretrain
except ImportError:
    from nets import get_model as models, load_pretrain

BATCH_SIZE = 128

parser = argparse.ArgumentParser()
parser.add_argument("-ckpt", "--checkpoint", type=str, required=True)
parser.add_argument("-i", "--input-json", type=str, required=True)
parser.add_argument("-o", "--output-json", type=str, required=True)
parser.add_argument("-m", "--margin", type=str, default=0)
parser.add_argument("--face-detector", action="store_true", help="Use face detector")


def get_features(model, input):
    features, logits = model(input)
    return features


def get_model(**kwargs) -> Module:
    net = models("resnet50")
    net._arch = "resnet50"

    if kwargs.get("path"):
        net = load_pretrain(kwargs["path"], net, getLogger(__name__))

    return net


class DatasetGenerator(Dataset):
    def __init__(self, data: List[Any]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def transform_image(self, fname: str) -> Any:
        center_crop = T.Compose(
            [
                T.ToTensor(),
                T.CenterCrop((500, 500)),
                T.Resize((256, 256)),
                T.CenterCrop((224, 224)),
                T.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),  # RGB [0,255] input, RGB normalize output
            ]
        )

        transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize((256, 256)),
                T.CenterCrop((224, 224)),
                T.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),  # RGB [0,255] input, RGB normalize output
            ]
        )

        img = None
        try:
            rimg = Image.open(fname).convert("RGB")
            rimg = np.array(rimg)
            h, w, c = rimg.shape
            # if h < w:
            #     rimg = np.rot90(rimg).copy()
            #     h, w, c = rimg.shape

            if h > 500 and w > 500:
                img = center_crop(rimg)
                return img, fname
            else:
                img = rimg
                img = transform(img)
                return img, fname

        except ValueError as e:
            print(f"Error in file:  {fname}")
            print(f"Error: {e}")
            traceback.print_exc()
            return None, "error-{}".format(fname)

    def __getitem__(self, index) -> Any:
        datapoint = self.data[index]
        return self.transform_image(datapoint)


def collate_fn(batch):
    x, f = [], []
    for item in batch:
        if item[0] is not None:
            x.append(item[0])
        else:
            x.append(torch.zeros(3, 224, 224))

        f.append(item[1])
    return torch.stack(x, dim=0), f


def get_scores(files: List[str], model: Module, position=0) -> List[Tuple[str, float]]:
    result: List[Tuple[str, float]] = []
    model.eval()
    model.cuda()
    wrapper = DatasetGenerator(files)
    for x, fnames in tqdm(
        DataLoader(
            wrapper,
            batch_size=BATCH_SIZE,
        ),
        position=position,
    ):
        x = x.cuda()
        if model._arch.startswith("mobilenet") or model._arch.startswith("shufflenet"):
            outputs = model(x)
        else:
            _, outputs = model(x)

        pred = F.softmax(outputs, dim=1)[:, 0].detach().cpu().numpy()
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
