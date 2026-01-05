import torch
from pathlib import Path
from logging import getLogger
from inference import get_model, get_scores
import os


def driver() -> None:
    log = getLogger()
    ckpts = [
        "resnet18_pICM2O_best.pth",
        "resnet18_pOCI2M_best.pth",
        "resnet18_pOCM2I_best.pth",
        "resnet18_pOMI2C_best.pth",
    ]
    paths = [str(p) for p in Path("/root/datasets/OCIM/Oulu/faces/Attack/Print/1_oulu_Train_files/1_1_01_2/").glob(
        "*.png"
    )]
    for ckpt in ckpts:
        print("Loading:", ckpt)
        model = get_model(path=f"../pretrained_models/GACD_FAS/{ckpt}")
        model.cuda().eval()
        results = get_scores(paths, model)
        print(sorted([x[1] for x in results]))


if __name__ == "__main__":
    driver()
