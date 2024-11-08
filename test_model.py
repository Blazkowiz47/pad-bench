import argparse
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models import get_model, get_score_function, get_transform_function
from util.logger import get_logger
from util import DatasetGenerator
from eval_loop import MODELS_CHECKPOINTS

log = get_logger("./logs/test.log")


def driver(model_name: str, path: str) -> None:
    log.info(f"Evaluating: {model_name}")
    real_path = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/3D_PAD_Datasets/2D_Face_Databases_PAD/iPhone11/Data_Split/real/test/IMG_1858.JPG"
    attack_path = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/3D_PAD_Datasets/2D_Face_Databases_PAD/iPhone11/Data_Split/attack/display/test/IMG_7263_1.jpg"
    transform_image = get_transform_function(model_name)

    def transform(img):
        return transform_image(img), torch.tensor([1, 0] if "real" in img else [0, 1])

    ds = DataLoader(DatasetGenerator([real_path, attack_path], transform), batch_size=2)
    model = get_model(
        model_name,
        {"arch": "resnet50", "n_classes": 2},
        log,
        path=path,
    )
    model.cuda().eval()

    get_scores = get_score_function(model_name)
    res = get_scores(ds, model, log)
    log.info(f"{res}")
    time.sleep(1)


if __name__ == "__main__":
    for model in MODELS_CHECKPOINTS:
        if model == "GACD_FAS":
            for protocol, ckpt in MODELS_CHECKPOINTS[model].items():
                driver(model, ckpt)
