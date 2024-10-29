"""
Main training file.
calls the train pipeline with configs.
"""

import argparse
import os

import numpy as np
import yaml

from cdatasets import get_dataset

# Incase you use wandb uncomment following line
# import wandb
from models import get_model, get_score_function, get_transform_function
from util import calculate_eer, initialise_dirs, logger

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
    "--rdir",
    type=str,
    default="/cluster/nbl-users/Shreyas-Sushrut-Raghu/3D_PAD_Datasets/",
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
    default=None,
    help="Load initial weights from partially/pretrained model.",
)

parser.add_argument(
    "-edir",
    "--evaluation-dir",
    type=str,
    default=None,
    help="Load initial weights from partially/pretrained model.",
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

    model = get_model(args.model, config, log, path=args.continue_model).to(device)
    log.info(str(model))
    wrapper = get_dataset(
        args.dataset, config, log, transform=get_transform_function(args.model)
    )

    get_scores = get_score_function(args.model)
    testds = wrapper.get_split("test")

    result = get_scores(testds, model)

    if "real" in result:
        np.savetxt(os.path.join(evaluation_dir, "real.txt"), np.array(result["real"]))
    if "attack" in result:
        np.savetxt(
            os.path.join(evaluation_dir, "attack.txt"), np.array(result["attack"])
        )

    if "real" in result and "attack" in result:
        eer = calculate_eer(result["real"], result["attack"])
        log.info(f"D-EER: {eer:.4f}")


if __name__ == "__main__":
    main()
