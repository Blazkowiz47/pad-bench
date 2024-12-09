"""
Main training file.
calls the train pipeline with configs.
"""

import argparse
import os

import numpy as np
import yaml

from cdatasets import get_dataset
from models import get_model, get_score_function, get_transform_function
from util import calculate_eer, initialise_dirs, logger, SOTA

parser = argparse.ArgumentParser(
    description="Evaluation Config",
    add_help=True,
)

parser.add_argument(
    "-m",
    "--model",
    default="DGUA_FAS",
    type=str,
    choices=[sota.name for sota in SOTA],
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
    default="*",
    type=str,
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

parser.add_argument(
    "--logger-level",
    type=str,
    choices=["INFO", "DEBUG", "CRITICAL", "ERROR"],
    default="INFO",
)


# You can add any additional arguments if you need here.
def only_calculate_eer():
    """
    Wrapper for the calculating only eer.
    """
    args = parser.parse_args()

    evaluation_dir = args.evaluation_dir
    initialise_dirs(evaluation_dir)

    logfile = os.path.join(evaluation_dir, "eval.log")
    log = logger.get_logger(logfile, args.logger_level)

    for ssplit in ["test"]:
        log.info(f"Evaluating {ssplit} split")
        real = np.loadtxt(
            os.path.join(evaluation_dir, f"{ssplit}_real.txt"),
        )
        attack = np.loadtxt(
            os.path.join(evaluation_dir, f"{ssplit}_attack.txt"),
        )
        maxi = max(np.max(attack), np.max(real))
        #         eer = calculate_eer(real, attack, reverse=True)
        #         log.info(
        #             f"D-EER {args.rdir.split('/')[-2]} {args.attack} ({ssplit}): {eer:.4f}"
        #         )
        attack = maxi - attack
        real = maxi - real

        np.savetxt(
            os.path.join(evaluation_dir, f"{ssplit}_real.txt"),
            real,
        )
        np.savetxt(
            os.path.join(evaluation_dir, f"{ssplit}_attack.txt"),
            attack,
        )


#         eer = calculate_eer(real, attack)
#         log.info(
#             f"D-EER {args.rdir.split('/')[-2]} {args.attack} ({ssplit}): {eer:.4f}"
#         )


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

    get_scores = get_score_function(sota)

    for ssplit in ["test", "train"]:
        splitds = wrapper.get_split(ssplit)
        log.info(f"Evaluating {ssplit} split")

        result = get_scores(splitds, model, log)

        if "real" in result:
            log.debug(f"Real scores: {len(result['real'])}")
            np.savetxt(
                os.path.join(evaluation_dir, f"{ssplit}_real.txt"),
                np.array(result["real"]),
            )
        if "attack" in result:
            log.debug(f"Attack scores: {len(result['attack'])}")
            np.savetxt(
                os.path.join(evaluation_dir, f"{ssplit}_attack.txt"),
                np.array(result["attack"]),
            )

        if "real" in result and "attack" in result:
            eer = calculate_eer(result["real"], result["attack"])
            log.info(f"D-EER ({ssplit}): {eer:.4f}")


if __name__ == "__main__":
    only_calculate_eer()
#     main()
