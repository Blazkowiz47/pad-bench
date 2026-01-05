# Real -> 0 Single class
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
    from .model.mfad import FAD_HAM_Net
except ImportError:
    from model.mfad import FAD_HAM_Net

BATCH_SIZE = 64
USE_FACE_DETECTOR = False


def get_model(**kwargs) -> Module:
    net: Module = FAD_HAM_Net(pretrain=False, variant="resnet50")
    if kwargs.get("path"):
        print(f"Loading weights from {kwargs['path']}")
        net.load_state_dict(torch.load(kwargs["path"], weights_only=True))
    return net


def get_features(model, input):
    x0 = input  # rgb image 3, 224, 224
    fad = model.FAD_Head(input)  # frequency: 12, 224, 224
    fad, fad_3, fad_2, fad_1 = model.FAD_encoder(fad)
    rgb, rgb_3, rgb_2, rgb_1 = model.RGB_encoder(x0)

    concate_3 = torch.cat((fad_3, rgb_3), dim=1)
    concate_2 = torch.cat((fad_2, rgb_2), dim=1)
    concate_1 = torch.cat((fad_1, rgb_1), dim=1)
    # high-level feature map using channel attention
    att_x_3 = model.channel_attention(concate_3)  # 14x14, 3
    att_x_3_14x14 = model.downsample(att_x_3)

    # low-level feature map using spatial attention
    att_x_2 = model.spa_1(concate_2)  # 28x28, 5
    att_x_2_14x14 = model.downsample(att_x_2)

    att_x_1 = model.spa_2(concate_1)  # 56x56, 7
    att_x_1_14x14 = model.downsample(att_x_1)

    x_concate = torch.cat((att_x_1_14x14, att_x_2_14x14, att_x_3_14x14), dim=1)

    map_x = model.lastconv1(x_concate)
    map_x = map_x.squeeze(1)

    x = torch.cat((fad, rgb), dim=1)  # concatenate last output feature
    x = model.linear1(x)
    x = model.bn(x)
    return x


class DatasetGenerator(Dataset):
    def __init__(self, data: List[Any]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def transform_image(self, fname: str) -> Any:
        transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize((256, 256)),
                T.CenterCrop((224, 224)),
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
            prefetch_factor=10,
        ),
        position=position,
    ):
        x = x.cuda()
        outputs, _ = model(x)
        pred = torch.sigmoid(outputs).detach().cpu().numpy()[:, 0]
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
