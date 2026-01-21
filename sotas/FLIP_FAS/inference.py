# 0 -> Fake
# 1 -> Real

import argparse
import json
from typing import Any, List

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torch.multiprocessing import set_start_method
from torchvision import transforms as T
from tqdm import tqdm

try:
    set_start_method("spawn", force=True)
except RuntimeError:
    pass
ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from .fas import flip_mcl
except ImportError:
    from fas import flip_mcl

BATCH_SIZE = 128


def get_model(**kwargs) -> Module:
    net = flip_mcl(
        in_dim=512, ssl_mlp_dim=4096, ssl_emb_dim=256
    )  # ssl applied to image, and euclidean distance applied to image and text cosine similarity
    if kwargs.get("ckpt"):
        net_ = torch.load(kwargs["ckpt"], weights_only=False)
        net.load_state_dict(net_["state_dict"])
    return net


class DatasetGenerator(Dataset):
    def __init__(self, data: List[Any]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def transform_image(self, fname: str) -> Any:
        transform = T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        img = Image.open(fname).convert("RGB")
        img = transform(img)
        return img, fname

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


def get_scores(files: List[str], model: Module, position=0) -> List[float]:
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
        cls_out, _ = model.forward_eval(x)
        pred = F.softmax(cls_out, dim=1).detach().cpu().numpy()[:, 1]
        result.extend([(fname, p) for fname, p in zip(fnames, pred.tolist())])

    return result


def get_features(model, input):
    image_features = model.model.encode_image(input)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    return image_features


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
