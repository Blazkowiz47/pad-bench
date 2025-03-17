import argparse
import json
from logging import Logger
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.nn import Module
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    from .fas import flip_mcl
except ImportError:
    from fas import flip_mcl

parser = argparse.ArgumentParser()
parser.add_argument("-ckpt", "--checkpoint", type=str, required=True)
parser.add_argument("-i", "--input-json", type=str, required=True)
parser.add_argument("-o", "--otput-json", type=str, required=True)


def get_model(ckpt: str = "") -> Module:
    net = flip_mcl(
        in_dim=512, ssl_mlp_dim=4096, ssl_emb_dim=256
    )  # ssl applied to image, and euclidean distance applied to image and text cosine similarity
    if ckpt:
        net_ = torch.load(ckpt)
        net.load_state_dict(net_["state_dict"])
    return net


class DatasetGenerator(Dataset):
    def __init__(self, data: List[Any]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def transform_image(self, fname: str) -> torch.Tensor:
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        try:
            img = Image.open(fname).resize((224, 224))
        except Exception as e:
            print(f"Error in file:  {fname}")
            print(f"Error: {e}")
            raise e

        return transform(img)

    def __getitem__(self, index) -> Any:
        datapoint = self.data[index]
        return self.transform_image(datapoint)


def get_scores(files: List[str], model: Module, position=0) -> List[float]:
    result: List[float] = []
    model.eval()
    model.cuda()
    for x in tqdm(
        DataLoader(DatasetGenerator(files), batch_size=64), position=position
    ):
        x = x.cuda()
        cls_out, _ = model.forward_eval(x)
        pred = F.softmax(cls_out, dim=1).detach().cpu().numpy()[:, 1]
        result.extend(pred.tolist())

    return result


def driver(ckpt: str, input_json: str, otput_json: str) -> None:
    with open(input_json, "r") as fp:
        input = json.load(fp)
    files = input["files"]
    model = get_model(ckpt)
    result = get_scores(files, model)
    with open(otput_json, "w+") as fp:
        json.dump({"result": result}, fp)


if __name__ == "__main__":
    args = parser.parse_args()
    driver(args.checkpoint, args.input_json, args.otput_json)
