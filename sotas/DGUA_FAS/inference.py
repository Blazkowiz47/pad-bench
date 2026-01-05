# Fake -> 0
# Real  -> 1
import argparse
import json
import os
from typing import Any, List

import cvnets
import torch
import torch.nn.functional as F
from PIL import ImageFile
from torch.multiprocessing import set_start_method
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    set_start_method("spawn", force=True)
except RuntimeError:
    pass
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from .model import transform_image as transform_image_
    from .option import get_training_arguments
except ImportError:
    from model import transform_image as transform_image_
    from option import get_training_arguments

BATCH_SIZE = 64
USE_FACE_DETECTOR = False


class DatasetGenerator(Dataset):
    def __init__(self, data: List[Any]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def transform_image(self, fname: str) -> Any:
        return transform_image_(fname), fname

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


def get_model(**kwargs):
    path = "/root/code/configs/mobilevit_s.yaml"
    if not os.path.exists("configs/mobilevit_s.yaml"):
        path = "./models/DGUA_FAS/configs/mobilevit_s.yaml"

    opts = get_training_arguments(config_path=path)
    net = cvnets.get_model(opts)

    if not isinstance(net, Module):
        raise ValueError("Cannot load DGUA_FAS model")

    if kwargs.get("path", ""):
        print("Loading weights:", kwargs["path"])
        net_ = torch.load(kwargs["path"], weights_only=False)
        net.load_state_dict(net_["state_dict"])

    return net


def get_features(model, input):
    x = model.extract_features(input)
    x = model.classifier[0](x)
    return x


def get_scores(
    files: List[str], model: Module, position=0
) -> List[float]:
    result: List[float] = []
    model.eval()
    model.cuda()
    wrapper = DatasetGenerator(files)
    for x, fnames in tqdm(
        DataLoader(
            wrapper,
            batch_size=BATCH_SIZE,
            num_workers=4,
            prefetch_factor=10,
        ),
        position=position,
    ):
        x = x.cuda()
        cls_out = model(x, True)[0]
        pred = F.softmax(cls_out, dim=1).detach().cpu().numpy()[:, 1]
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
