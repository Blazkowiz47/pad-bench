"""
Main training file.
calls the train pipeline with configs.
"""

import argparse
import os

import numpy as np
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from tqdm import tqdm
import yaml

from cdatasets import get_dataset
from models import (
    get_model,
    get_score_function,
    get_transform_function,
    get_finetune_epoch_step,
)
from util import SOTA, calculate_eer, initialise_dirs, logger

parser = argparse.ArgumentParser(
    description="Evaluation Config",
    add_help=True,
)

parser.add_argument(
    "-m",
    "--model",
    default="DGUA_FAS",
    type=str,
    help="Model name.",
)

parser.add_argument(
    "-c",
    "--config",
    default="configs/train.yaml",
    type=str,
    help="Train config file.",
)

parser.add_argument(
    "-d",
    "--dataset",
    default="standard",
    type=str,
    help="""
    Give a single dataset name or multiple datasets to chain together.
    eg: -d datset1 
    """,
)
parser.add_argument(
    "--attack",
    default="soft_plastic",
    type=str,
)

parser.add_argument(
    "--rdir",
    type=str,
    default="/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/3D_PAD_Datasets/2D_Face_Databases_PAD/iPhone11/Data_Split/",
    help="""
    Root directory of the dataset
    """,
)

parser.add_argument(
    "--batch-size",
    default=32,
    type=int,
    help="Override batch-size fetched from the config file.",
)

parser.add_argument(
    "-ckpt",
    "--continue-model",
    type=str,
    default="./pretrained_models/DGUA_FAS/I&C&MtoO/best_model.pth.tar",
    help="Load initial weights from partially/pretrained model.",
)

parser.add_argument(
    "-edir",
    "--evaluation-dir",
    type=str,
    default="./tmp/DGUA_FAS/icmo/finetune/indian_pad/iPhone11/soft_plastic",
    help="Load initial weights from partially/pretrained model.",
)

parser.add_argument(
    "--logger-level",
    type=str,
    choices=["INFO", "DEBUG", "CRITICAL", "ERROR"],
    default="DEBUG",
)

parser.add_argument(
    "--epochs",
    default=100,
    type=int,
)

# You can add any additional arguments if you need here.


def main():
    """
    Wrapper for the driver.
    """
    args = parser.parse_args()

    with open(args.config, "r") as fp:
        config = yaml.safe_load(fp)

    evaluation_dir = args.evaluation_dir
    initialise_dirs(evaluation_dir)
    logfile = os.path.join(evaluation_dir, "eval.log")
    log = logger.get_logger(logfile, args.logger_level)

    device = "cuda"  # You can change this to cpu.
    if "resnet18_pOMI2C_best" in args.continue_model:
        config["n_classes"] = 2
    sota = SOTA(args.model)
    model = get_model(sota, config, log, path=args.continue_model).to(device)
    log.info(str(model))
    if "iPhone" in args.rdir:
        config["facedetect"] = "Face_Detect"

    wrapper = get_dataset(
        args.dataset,
        config,
        log,
        rdir=args.rdir,
        transform=get_transform_function(sota),
        attack=args.attack,
    )

    fine_tune_epoch_caller = get_finetune_epoch_step(sota)
    get_scores = get_score_function(sota)
    model.train()
    optimizer = SGD(
        [p for p in model.parameters() if p.requires_grad],
        float(config.get("lr", 1e-4)),
    )
    loss_fn = CrossEntropyLoss().to(device)

    for epoch in tqdm(range(args.epochs), position=0):
        trainds = wrapper.get_split(
            "train",
        )
        testds = wrapper.get_split("test")
        log.info(f"Finetuning epoch {epoch}:")
        fine_tune_epoch_caller(model, trainds, optimizer, loss_fn, log, device)

        result = get_scores(testds, model, log)
        #         if "real" in result:
        #             log.debug(f"Real scores: {len(result['real'])}")
        #             np.savetxt(
        #                 os.path.join(evaluation_dir, f"{ssplit}_real.txt"),
        #                 np.array(result["real"]),
        #             )
        #         if "attack" in result:
        #             log.debug(f"Attack scores: {len(result['attack'])}")
        #             np.savetxt(
        #                 os.path.join(evaluation_dir, f"{ssplit}_attack.txt"),
        #                 np.array(result["attack"]),
        #             )
        if "real" in result and "attack" in result:
            eer = calculate_eer(result["real"], result["attack"])
            log.info(f"D-EER for epoch {epoch}: {eer:.4f}")


if __name__ == "__main__":
    main()
