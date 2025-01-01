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
    "--dataset-name",
    default="idiap",
    choices=[
        "3D_mask_google",
        "HDA_flicker_attacks",
        "idiap",
        "test_dataset_pro",
        "training_pro_ibeta_level2",
    ],
    type=str,
)
parser.add_argument(
    "--attack",
    default="*",
    type=str,
)

parser.add_argument(
    "--rdir",
    type=str,
    default="/home/ubuntu/datasets/test",
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
    "--checkpoint",
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

    log.info(f"Evaluating {args.dataset_name} split")
    real = np.loadtxt(
        os.path.join(evaluation_dir, f"{args.dataset_name}_real.txt"),
    )
    attack = np.loadtxt(
        os.path.join(evaluation_dir, f"{args.dataset_name}_attack.txt"),
    )
    maxi = max(np.max(attack), np.max(real))
    #         eer = calculate_eer(real, attack, reverse=True)
    #         log.info(
    #             f"D-EER {args.rdir.split('/')[-2]} {args.attack} ({ssplit}): {eer:.4f}"
    #         )
    attack = maxi - attack
    real = maxi - real

    np.savetxt(
        os.path.join(evaluation_dir, f"{args.dataset_name}_real.txt"),
        real,
    )
    np.savetxt(
        os.path.join(evaluation_dir, f"{args.dataset_name}_attack.txt"),
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
    if "resnet18_pOMI2C_best" in args.checkpoint:
        config["n_classes"] = 2

    sota = SOTA(args.model)
    model = get_model(sota, config, log, path=args.checkpoint).to(device)
    log.info(str(model))
    if "iPhone" in args.rdir:
        config["facedetect"] = "Face_Detect"

    wrapper = get_dataset(
        args.dataset,
        config,
        log,
        rdir=os.path.join(args.rdir, args.dataset_name),
        transform=get_transform_function(sota),
        attack=args.attack,
    )

    get_scores = get_score_function(sota)

    dataset = wrapper.get(num_workers=8)

    result = get_scores(dataset, model, log)

    print(f"Evaluation of {args.dataset_name}")
    if "real" in result:
        print(f"Real scores: {len(result['real'])}")
        np.savetxt(
            os.path.join(evaluation_dir, f"{args.dataset_name}_real.txt"),
            np.array(result["real"]),
        )
    if "attack" in result:
        print(f"Attack scores: {len(result['attack'])}")
        np.savetxt(
            os.path.join(evaluation_dir, f"{args.dataset_name}_attack.txt"),
            np.array(result["attack"]),
        )

    if "real" in result and "attack" in result:
        eer, far, frr, thresholds = calculate_eer(result["real"], result["attack"])
        ida1 = np.argmin(np.abs(far - 1))
        ida3 = np.argmin(np.abs(far - 3))
        ida5 = np.argmin(np.abs(far - 5))

        idb1 = np.argmin(np.abs(frr - 1))
        idb3 = np.argmin(np.abs(frr - 3))
        idb5 = np.argmin(np.abs(frr - 5))
        print(f"D-EER: {eer:.4f}")
        print(
            f"BPCER (@APCER = 1%) = {frr[ida1]:.4f}@{thresholds[ida1]:.4f} APCER (@BPCER = 1%) = {far[idb1]:.4f}@{thresholds[idb1]:.4f}"
        )
        print(
            f"BPCER (@APCER = 3%) = {frr[ida3]:.4f}@{thresholds[ida3]:.4f} APCER (@BPCER = 3%) = {far[idb3]:.4f}@{thresholds[idb3]:.4f}"
        )
        print(
            f"BPCER (@APCER = 5%) = {frr[ida5]:.4f}@{thresholds[ida5]:.4f} APCER (@BPCER = 5%) = {far[idb5]:.4f}@{thresholds[idb5]:.4f}"
        )


if __name__ == "__main__":
    main()
#     only_calculate_eer()
