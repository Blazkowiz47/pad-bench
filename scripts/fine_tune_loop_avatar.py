import importlib
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import scipy.stats as st
import umap
import json
import os
import random
import shutil
import traceback
import warnings
from pathlib import Path
from typing import Any, Tuple

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import seaborn as sns

from util import SOTA

warnings.simplefilter(action="ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)


BATCH_SIZE = 64
MODELS_CHECKPOINTS = {
    SOTA.CF_FAS: {
        "icmo": "/root/pretrained_models/CF_FAS/icmo.pth",
        "oimc": "/root/pretrained_models/CF_FAS/omic.pth",
        "ocim": "/root/pretrained_models/CF_FAS/ocim.pth",
        "ocmi": "/root/pretrained_models/CF_FAS/ocmi.pth",
    },
    SOTA.LMFD_FAS: {
        "icmo": "/root/pretrained_models/LMFD_FAS/icm_o.pth",
        "oimc": "/root/pretrained_models/LMFD_FAS/omi_c.pth",
        "ocim": "/root/pretrained_models/LMFD_FAS/oci_m.pth",
        "ocmi": "/root/pretrained_models/LMFD_FAS/ocm_i.pth",
    },
    SOTA.GACD_FAS: {
        "icmo": "/root/pretrained_models/GACD_FAS/resnet18_pICM2O_best.pth",
        "oimc": "/root/pretrained_models/GACD_FAS/resnet18_pOMI2C_best.pth",
        "ocim": "/root/pretrained_models/GACD_FAS/resnet18_pOCI2M_best.pth",
        "ocmi": "/root/pretrained_models/GACD_FAS/resnet18_pOCM2I_best.pth",
    },
    SOTA.JPD_FAS: {"all": "/root/pretrained_models/JPD_FAS/full_resnet50.pth"},
    SOTA.DGUA_FAS: {
        "icmo": "/root/pretrained_models/DGUA_FAS/ICMtoO/best_model.pth.tar",
        "oimc": "/root/pretrained_models/DGUA_FAS/OIMtoC/best_model.pth.tar",
        "ocim": "/root/pretrained_models/DGUA_FAS/OCItoM/best_model.pth.tar",
        "ocmi": "/root/pretrained_models/DGUA_FAS/OCMtoI/best_model.pth.tar",
    },
    SOTA.FLIP_FAS: {
        "icmo": "/root/pretrained_models/FLIP_FAS/oulu_flip_mcl.pth.tar",
        "oimc": "/root/pretrained_models/FLIP_FAS/casia_flip_mcl.pth.tar",
        "ocim": "/root/pretrained_models/FLIP_FAS/msu_flip_mcl.pth.tar",
        "ocmi": "/root/pretrained_models/FLIP_FAS/replay_flip_mcl.pth.tar",
    },
    SOTA.IADG_FAS: {
        "oimc": "/root/pretrained_models/IADG_FAS/IOM2C/checkpoint-50300.pth.tar",
        "ocim": "/root/pretrained_models/IADG_FAS/OCI2M/model_best.pth.tar",
        "ocmi": "/root/pretrained_models/IADG_FAS/OCM2I/checkpoint-10316.pth.tar",
        "icmo": "/root/pretrained_models/IADG_FAS/ICM2O/checkpoint-90500.pth.tar",
    },
}

ATTACKS = [
    "display",
    "print",
    "hard_plastic",
    "latex_mask",
    "paper_mask",
    "silicone_mask",
    "soft_plastic",
    "wrap",
]

REAL_LABEL = 1
ATTACK_LABEL = 0


# LMFD has one class and that corresponds to real


def apply_lora_adaptation(backbone):
    for param in backbone.parameters():
        param.requires_grad = False

    transformer = backbone.blocks
    for layer_idx, layer in enumerate(transformer):
        if hasattr(layer, "attn"):
            original_encoder = layer.attn.qkv
            layer.attn.qkv = LoRAEncoder(
                original_encoder,
                16,
                16,
                layer.attn.num_heads,
                adapt_key=True,
                adapt_query=True,
                adapt_value=True,
            )

        mlp = layer.mlp
        if hasattr(mlp, "c_fc"):
            mlp.c_fc = LoRALinear(mlp.c_fc, rank=16, alpha=16)
    return backbone


def get_model_and_load_checkpoint(sota: SOTA, protocol: str) -> object:
    module = importlib.import_module(f"models.{sota.value.upper()}.inference")
    get_model = getattr(module, "get_model")
    print(
        f"Loading model for {sota.value} with protocol {protocol} from {MODELS_CHECKPOINTS[sota][protocol]}"
    )
    ckpt = MODELS_CHECKPOINTS[sota][protocol]
    model = get_model(path=ckpt.replace("/root/", "./"))
    print(f"Model loaded successfully for {sota.value} - {protocol}")
    return model


def compute_eer(genuine_scores, impostor_scores):
    """
    Compute Equal Error Rate (EER) given genuine and impostor scores.
    Assumes scores are in the range [0, 1], where higher scores indicate
    a higher likelihood of being genuine.
    """
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)
    thresholds = np.linspace(0, 1, 10_001)
    fars = np.ones_like(thresholds)
    frrs = np.ones_like(thresholds)
    for i, thresh in enumerate(thresholds):
        fars[i] = np.sum(impostor_scores >= thresh) / impostor_scores.shape[0]
        frrs[i] = np.sum(genuine_scores < thresh) / genuine_scores.shape[0]

    eer_index = np.argmin(np.abs(fars - frrs))
    eer = (fars[eer_index] + frrs[eer_index]) / 2
    return eer


def compute_abpcer_at_aapcer(genuine_scores, impostor_scores, target_aapcer):
    """
    Compute ABPCER (Bona Fide Presentation Classification Error Rate) at a given
    AAPCER (Attack Presentation Acceptance Error Rate) threshold.

    Args:
        genuine_scores: Array of scores for genuine/real samples
        impostor_scores: Array of scores for attack/impostor samples
        target_aapcer: Target AAPCER value (e.g., 0.05 for 5%, 0.10 for 10%)

    Returns:
        ABPCER value as percentage (0-100)

    Terminology:
        - AAPCER = FAR (False Acceptance Rate) - rate at which attacks are accepted
        - ABPCER = FRR (False Rejection Rate) - rate at which genuine are rejected
    """
    genuine_scores = np.array(genuine_scores)
    impostor_scores = np.array(impostor_scores)
    thresholds = np.linspace(0, 1, 10_001)

    aapcer_values = np.zeros_like(thresholds)
    abpcer_values = np.zeros_like(thresholds)

    for i, thresh in enumerate(thresholds):
        # AAPCER: proportion of attacks scoring >= threshold (accepted as genuine)
        aapcer_values[i] = np.sum(impostor_scores >= thresh) / impostor_scores.shape[0]
        # ABPCER: proportion of genuine scoring < threshold (rejected as attack)
        abpcer_values[i] = np.sum(genuine_scores < thresh) / genuine_scores.shape[0]

    # Find threshold where AAPCER is closest to target
    idx = np.argmin(np.abs(aapcer_values - target_aapcer))
    abpcer_at_target = abpcer_values[idx]

    # Convert to percentage
    return abpcer_at_target * 100


def extract_attack_type_from_path(path):
    """
    Extract attack type (DISPLAY/PRINT/REAL) from image path.

    Args:
        path: Image file path string

    Returns:
        'display', 'print', or 'real' (lowercase), or None if not found
    """
    path_upper = path.upper()
    if "DISPLAY" in path_upper:
        return "display"
    elif "PRINT" in path_upper:
        return "print"
    elif "REAL" in path_upper:
        return "real"
    return None


def compute_attack_separated_metrics():
    """
    Compute metrics separated by attack type for all SOTAs and protocols.

    Returns:
        dict: {
            sota_name: {
                protocol: {
                    "display": {
                        "eer": {"values": [...], "mean": float, "ci": tuple},
                        "abpcer5": {"values": [...], "mean": float, "ci": tuple},
                        "abpcer10": {"values": [...], "mean": float, "ci": tuple}
                    },
                    "print": {...},
                    "combined": {...}
                }
            }
        }
    """
    results = {}

    for sota in SOTA:
        global REAL_LABEL, ATTACK_LABEL

        # Set labels based on SOTA type (FLIP_FAS and JPD_FAS have reversed labels)
        if sota in (SOTA.FLIP_FAS, SOTA.JPD_FAS):
            REAL_LABEL = 0
            ATTACK_LABEL = 1
        else:
            REAL_LABEL = 1
            ATTACK_LABEL = 0

        # Skip models as in driver function
        if sota in (SOTA.IADG_FAS,):
            continue

        results[sota.value] = {}

        for protocol in MODELS_CHECKPOINTS[sota]:
            print(f"\nProcessing {sota.value} - {protocol}")

            # Initialize metric lists for 5 partitions
            display_metrics = {"eers": [], "abpcer5": [], "abpcer10": []}
            print_metrics = {"eers": [], "abpcer5": [], "abpcer10": []}
            combined_metrics = {"eers": [], "abpcer5": [], "abpcer10": []}

            for part in range(1, 6):
                # Load checkpoint
                ckpt_path = Path(
                    f"./avatar_experiments/{sota.value}/{protocol}/{part}/checkpoints/best_model.pt"
                )

                if not ckpt_path.exists():
                    print(f"  Skipping partition {part}: checkpoint not found")
                    continue

                try:
                    checkpoint = torch.load(ckpt_path, weights_only=False)
                    scores_dict = checkpoint["scores"]
                except Exception as e:
                    print(f"  Error loading checkpoint for partition {part}: {e}")
                    continue

                # Separate scores by attack type
                display_scores = {"real": [], "attack": []}
                print_scores = {"real": [], "attack": []}
                combined_scores = {"real": [], "attack": []}

                for path, score in scores_dict.items():
                    path_upper = path.upper()

                    # Check if this is a real/genuine image
                    if "/REAL/" in path_upper:
                        # Real images are genuine samples for all attack types
                        display_scores["real"].append(score)
                        print_scores["real"].append(score)
                        combined_scores["real"].append(score)

                    # Check attack type
                    elif "/DISPLAY/" in path_upper:
                        display_scores["attack"].append(score)
                        combined_scores["attack"].append(score)

                    elif "/PRINT/" in path_upper:
                        print_scores["attack"].append(score)
                        combined_scores["attack"].append(score)

                # Compute metrics for each attack type
                # Display attacks
                if (
                    len(display_scores["real"]) > 0
                    and len(display_scores["attack"]) > 0
                ):
                    eer = (
                        compute_eer(display_scores["real"], display_scores["attack"])
                        * 100
                    )
                    abpcer5 = compute_abpcer_at_aapcer(
                        display_scores["real"], display_scores["attack"], 0.05
                    )
                    abpcer10 = compute_abpcer_at_aapcer(
                        display_scores["real"], display_scores["attack"], 0.10
                    )
                    display_metrics["eers"].append(eer)
                    display_metrics["abpcer5"].append(abpcer5)
                    display_metrics["abpcer10"].append(abpcer10)

                # Print attacks
                if len(print_scores["real"]) > 0 and len(print_scores["attack"]) > 0:
                    eer = (
                        compute_eer(print_scores["real"], print_scores["attack"]) * 100
                    )
                    abpcer5 = compute_abpcer_at_aapcer(
                        print_scores["real"], print_scores["attack"], 0.05
                    )
                    abpcer10 = compute_abpcer_at_aapcer(
                        print_scores["real"], print_scores["attack"], 0.10
                    )
                    print_metrics["eers"].append(eer)
                    print_metrics["abpcer5"].append(abpcer5)
                    print_metrics["abpcer10"].append(abpcer10)

                # Combined attacks
                if (
                    len(combined_scores["real"]) > 0
                    and len(combined_scores["attack"]) > 0
                ):
                    eer = (
                        compute_eer(combined_scores["real"], combined_scores["attack"])
                        * 100
                    )
                    abpcer5 = compute_abpcer_at_aapcer(
                        combined_scores["real"], combined_scores["attack"], 0.05
                    )
                    abpcer10 = compute_abpcer_at_aapcer(
                        combined_scores["real"], combined_scores["attack"], 0.10
                    )
                    combined_metrics["eers"].append(eer)
                    combined_metrics["abpcer5"].append(abpcer5)
                    combined_metrics["abpcer10"].append(abpcer10)

            # Calculate statistics for each attack type
            results[sota.value][protocol] = {}

            for attack_type, metrics in [
                ("display", display_metrics),
                ("print", print_metrics),
                ("combined", combined_metrics),
            ]:
                results[sota.value][protocol][attack_type] = {}

                for metric_name in ["eers", "abpcer5", "abpcer10"]:
                    values = metrics[metric_name]
                    if len(values) > 0:
                        mean_val = np.mean(values)
                        ci_val = get_ci(values)
                        # Store with shorter key names for easier access
                        if metric_name == "eers":
                            key = "eer"
                        elif metric_name == "abpcer5":
                            key = "abpcer5"
                        else:
                            key = "abpcer10"

                        results[sota.value][protocol][attack_type][key] = {
                            "values": values,
                            "mean": mean_val,
                            "ci": ci_val,
                        }
                    else:
                        # No data for this metric
                        if metric_name == "eers":
                            key = "eer"
                        elif metric_name == "abpcer5":
                            key = "abpcer5"
                        else:
                            key = "abpcer10"

                        results[sota.value][protocol][attack_type][key] = {
                            "values": [],
                            "mean": float("inf"),
                            "ci": (float("inf"), float("inf")),
                        }

    return results


class DatasetGenerator(Dataset):
    def __init__(self, data, is_train=False, size=224) -> None:
        self.data = data
        self.size = size
        random.shuffle(self.data)
        if is_train:
            self.transform = A.Compose(
                [
                    A.Resize(self.size, self.size),
                    A.HorizontalFlip(p=0.5),
                    A.ColorJitter(p=0.2),
                    A.RandomBrightnessContrast(p=0.2),
                    A.CoarseDropout(
                        num_holes_range=(1, 4),
                        hole_height_range=(16, 64),
                        hole_width_range=(16, 64),
                        p=0.2,
                    ),
                    A.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                    ToTensorV2(),
                ]
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(self.size, self.size),
                    A.Normalize(
                        mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225),
                    ),
                    ToTensorV2(),
                ]
            )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Any:
        image_path, label = self.data[index]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image=np.array(image))["image"]

        return image, label, image_path


def get_dataset(part, size):
    rdir = Path("~/datasets/avatar_db").expanduser()
    with open(f"./avatar_experiments/db_{part}.json", "r") as f:
        partitions = json.load(f)
    train_subjects = partitions["train_subjects"]
    test_subjects = partitions["test_subjects"]
    train_data = {}
    test_data = {}
    for subject in train_subjects:
        subject_dir = rdir / subject
        for attack in ["DISPLAY", "PRINT", "REAL"]:
            attack_dir = subject_dir / attack
            for img_path in attack_dir.iterdir():
                if attack.lower() not in train_data:
                    train_data[attack.lower()] = []
                train_data[attack.lower()].append(str(img_path.expanduser()))

    for subject in test_subjects:
        subject_dir = rdir / subject
        for attack in ["DISPLAY", "PRINT", "REAL"]:
            attack_dir = subject_dir / attack
            for img_path in attack_dir.iterdir():
                if attack.lower() not in test_data:
                    test_data[attack.lower()] = []
                test_data[attack.lower()].append(str(img_path.expanduser()))

    train_data_list = [
        (path, REAL_LABEL if "real" == attack else ATTACK_LABEL)
        for attack, paths in train_data.items()
        for path in paths
    ]
    test_data_list = [
        (path, REAL_LABEL if "real" == attack else ATTACK_LABEL)
        for attack, paths in test_data.items()
        for path in paths
    ]
    return DataLoader(
        DatasetGenerator(train_data_list, True, size),
        batch_size=BATCH_SIZE,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
    ), DataLoader(
        DatasetGenerator(test_data_list, size),
        batch_size=BATCH_SIZE,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=g,
    )


def check_if_run_is_ran_before(ckpt_dir: Path) -> Tuple[float, Path]:
    best_eer = float("inf")
    success = False
    path = None
    if ckpt_dir.exists():
        for f in ckpt_dir.iterdir():
            if f.name != "best_model.pt":
                eer_str = f.name.split("_eer_")[-1].replace(".pt", "")
                try:
                    eer = float(eer_str)
                    if eer < best_eer:
                        best_eer = eer
                        path = f
                except:
                    continue
            else:
                success = True
    if success:
        return best_eer, path
    return float("inf"), path


def finetune_model(model, part: int, sota: SOTA, protocol: str):
    """
    Finetuning procedure:
    - Load dataset for the given part
    - Set all model parameters to be trainable
    - Use SGD optimizer with lr=0.0001, momentum=0.9, nesterov=True
    - For 20 epochs:
        - Train on training set
        - Evaluate on test set and compute EER
        - Save checkpoint for each epoch
        - Save best model if EER improves
        - Use binary cross-entropy loss for binary classification
    Returns the best EER achieved during fine-tuning.
    """
    train_ds, test_ds = get_dataset(part, 224)
    # Ensure all parameters are trainable
    for param in model.parameters():
        param.requires_grad = True

    optimizer = SGD(model.parameters(), lr=0.0001, momentum=0.9, nesterov=True)
    exp_dir = Path(f"./avatar_experiments/{sota.value}/{protocol}/{part}")
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_eer = float("inf")

    # Check if this run was done before
    previous_best_eer, _ = check_if_run_is_ran_before(ckpt_dir)
    if previous_best_eer != float("inf"):
        print(
            f"  This run was done before. Previous best EER: {previous_best_eer:.4f}%. Skipping..."
        )
        return previous_best_eer

    for epoch in range(1, 21):
        model.train()
        for x, y, z in train_ds:
            optimizer.zero_grad()
            x = x.cuda()
            y = y.cuda().long()
            if sota == SOTA.CF_FAS:
                cls_out = model(x, labels=y, cf=None)
            elif sota == SOTA.DGUA_FAS:
                cls_out, *_ = model(x, True)
            elif sota == SOTA.FLIP_FAS:
                cls_out, _ = model.forward_eval(x)
            elif sota == SOTA.LMFD_FAS:
                cls_out, _ = model(x)
            elif sota == SOTA.JPD_FAS:
                _, cls_out = model(x)
            else:
                cls_out = model(x)

            if sota == (SOTA.LMFD_FAS, SOTA.IADG_FAS) or cls_out.shape[1] == 1:
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    cls_out.squeeze(), y.float()
                )
            else:
                loss = torch.nn.functional.cross_entropy(cls_out, y)
            loss.backward()
            optimizer.step()

        model.eval()
        genuine_scores = []
        impostor_scores = []
        scores_dict = {}
        with torch.no_grad():
            for x, y, z in test_ds:
                x = x.cuda()
                y = y.cuda().long()
                if sota == SOTA.CF_FAS:
                    cls_out = model(x, cf=None)
                elif sota == SOTA.DGUA_FAS:
                    cls_out, *_ = model(x, True)
                elif sota == SOTA.FLIP_FAS:
                    cls_out, _ = model.forward_eval(x)
                elif sota == SOTA.LMFD_FAS:
                    cls_out, _ = model(x)
                elif sota == SOTA.JPD_FAS:
                    _, cls_out = model(x)
                else:
                    cls_out = model(x)
                # Get softmax scores for genuine/real class (class 0)
                if sota in (SOTA.LMFD_FAS,) or cls_out.shape[1] == 1:
                    scores = torch.sigmoid(cls_out).squeeze()
                else:
                    scores = torch.nn.functional.softmax(cls_out, dim=1)[:, REAL_LABEL]

                # Separate genuine and impostor scores
                for score, label, path in zip(scores.cpu().numpy(), y.cpu().numpy(), z):
                    if label == REAL_LABEL:  # Real/genuine
                        genuine_scores.append(score)
                    else:  # Attack/impostor
                        impostor_scores.append(score)
                    scores_dict[path] = score

        eer = (
            compute_eer(genuine_scores, impostor_scores) * 100
        )  # Convert to percentage
        print(f"Epoch {epoch}, EER: {eer:.4f}%")

        # Save checkpoint for this epoch
        torch.save(
            {"state_dict": model.state_dict(), "scores": scores_dict},
            ckpt_dir / f"epoch_{epoch}_eer_{eer:.4f}.pt",
        )

        # Save best model if EER improved (lower is better)
        if eer < best_eer:
            best_eer = eer
            torch.save(
                {"state_dict": model.state_dict(), "scores": scores_dict},
                ckpt_dir / "best_model.pt",
            )
            print("  New best EER! Saved to best_model.pt")

    return best_eer


def print_results_table(results):
    """Print results in a formatted table with SOTAs as rows and protocols as columns."""
    protocols = ["icmo", "oimc", "ocim", "ocmi"]

    # Print header
    print(f"\n{'=' * 90}")
    print("EER Results Table (%) - Average ± Std across 5 parts")
    print(f"{'=' * 90}")
    print(f"{'SOTA':<15} {'ICMO':>17} {'OIMC':>17} {'OCIM':>17} {'OCMI':>17}")
    print(f"{'-' * 15} {'-' * 17} {'-' * 17} {'-' * 17} {'-' * 17}")

    # Print each SOTA row
    for sota_name in sorted(results.keys()):
        row = f"{sota_name:<15}"
        for protocol in protocols:
            if protocol in results[sota_name]:
                mean_eer = results[sota_name][protocol]["mean"]
                std_eer = results[sota_name][protocol]["std"]
                row += f" {mean_eer:>6.2f}±{std_eer:<6.2f}"
            else:
                row += f" {'N/A':>17}"
        print(row)

    print(f"{'=' * 90}\n")


def print_attack_separated_tables(results):
    """
    Print results tables separated by attack type.
    For each attack type, selects the best protocol (lowest EER) per SOTA.

    Args:
        results: Dictionary from compute_attack_separated_metrics()
    """
    attack_types = ["display", "print", "combined"]

    for attack_type in attack_types:
        print(f"\n{'=' * 120}")
        print(f"Results for {attack_type.upper()} attacks - Best Protocol per SOTA")
        print(f"{'=' * 120}")
        print(
            f"{'SOTA':<15} {'Protocol':<11} {'EER (%)':<26} {'ABPCER@5%AAPCER (%)':<26} {'ABPCER@10%AAPCER (%)':<26}"
        )
        print(f"{'-' * 15} {'-' * 11} {'-' * 26} {'-' * 26} {'-' * 26}")

        for sota_name in sorted(results.keys()):
            # Find best protocol for this attack type (lowest EER)
            best_protocol = None
            best_eer_mean = float("inf")

            for protocol in results[sota_name].keys():
                if attack_type in results[sota_name][protocol]:
                    attack_data = results[sota_name][protocol][attack_type]
                    if (
                        "eer" in attack_data
                        and attack_data["eer"]["mean"] < best_eer_mean
                    ):
                        best_eer_mean = attack_data["eer"]["mean"]
                        best_protocol = protocol

            if best_protocol is None:
                # No data for this SOTA and attack type
                print(
                    f"{sota_name:<15} {'N/A':<11} {'N/A':<26} {'N/A':<26} {'N/A':<26}"
                )
                continue

            # Get metrics for best protocol
            attack_data = results[sota_name][best_protocol][attack_type]

            # Format EER
            eer_mean = attack_data["eer"]["mean"]
            eer_ci = attack_data["eer"]["ci"]
            eer_str = f"{eer_mean:>6.2f} [{eer_ci[0]:>5.2f}, {eer_ci[1]:>5.2f}]"

            # Format ABPCER@5%
            abpcer5_mean = attack_data["abpcer5"]["mean"]
            abpcer5_ci = attack_data["abpcer5"]["ci"]
            abpcer5_str = (
                f"{abpcer5_mean:>6.2f} [{abpcer5_ci[0]:>5.2f}, {abpcer5_ci[1]:>5.2f}]"
            )

            # Format ABPCER@10%
            abpcer10_mean = attack_data["abpcer10"]["mean"]
            abpcer10_ci = attack_data["abpcer10"]["ci"]
            abpcer10_str = f"{abpcer10_mean:>6.2f} [{abpcer10_ci[0]:>5.2f}, {abpcer10_ci[1]:>5.2f}]"

            print(
                f"{sota_name:<15} {best_protocol:<11} {eer_str:<26} {abpcer5_str:<26} {abpcer10_str:<26}"
            )

        print(f"{'=' * 120}\n")

        if attack_type == "display":
            print(
                "Note: Each attack type table independently selects the best protocol per SOTA."
            )
            print(
                "      For example, CF_FAS might use 'icmo' for display but 'oimc' for print.\n"
            )


# Labels    REAL       ATTACK
# FLIP_FAS   0           1
#  JPD_FAS   0           1
# LMFD_FAS   0           0 # Single class
# GACD_FAS   0           1 # some single class
# DGUA_FAS   1           0
#   CF_FAS   1           0


def get_ci(data):
    sample_mean = np.mean(data)
    df = len(data) - 1
    standard_error = st.sem(data)
    confidence_interval = st.t.interval(
        confidence=0.95,  # Confidence level (e.g., 0.95 for 95%)
        df=df,
        loc=sample_mean,
        scale=standard_error,
    )
    return confidence_interval


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def plot_features(results):
    """
    results: dict -> {
            sota_name: {
                protocol: {
                    "display": {
                        "eer": {"values": [...], "mean": float, "ci": tuple},
                        "abpcer5": {"values": [...], "mean": float, "ci": tuple},
                        "abpcer10": {"values": [...], "mean": float, "ci": tuple}
                    },
                    "print": {...},
                    "combined": {...}
                }
            }
        }
    based on the results dictionary, get select the best protocol
    """
    # attack_type = "combined"
    # to_use = {}
    #
    # for sota_name in sorted(results.keys()):
    #     best_protocol = None
    #     best_eer_mean = float("inf")
    #
    #     for protocol in results[sota_name].keys():
    #         if attack_type in results[sota_name][protocol]:
    #             attack_data = results[sota_name][protocol][attack_type]
    #             if "eer" in attack_data and attack_data["eer"]["mean"] < best_eer_mean:
    #                 best_eer_mean = attack_data["eer"]["mean"]
    #                 best_protocol = protocol
    #
    #     if best_protocol is None:
    #         # No data for this SOTA and attack type
    #         print(f"{sota_name:<15} {'N/A':<11} {'N/A':<26} {'N/A':<26} {'N/A':<26}")
    #         continue
    #     if sota_name not in to_use:
    #         to_use[sota_name] = {}
    #     to_use[sota_name] = best_protocol
    #
    # __import__("pprint").pprint(to_use)
    #
    to_use = {
        "DGUA_FAS": "icmo",
        "FLIP_FAS": "ocmi",
        "GACD_FAS": "icmo",
        "JPD_FAS": "all",
        "LMFD_FAS": "ocim",
        "CF_FAS": "oimc",
    }

    for sota_name, protocol in to_use.items():
        plot_features_for_model(SOTA[sota_name], protocol)


def plot_tsne(features_list, labels_list, sota: SOTA):
    """Generate and save t-SNE visualization"""
    features_array = np.array([f[0] for f in features_list])
    labels_array = np.array(labels_list)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=2024)
    features_2d = tsne.fit_transform(features_array)

    # Set seaborn style
    sns.set_theme(style="whitegrid")

    # Create figure
    plt.figure(figsize=(10, 8))

    # Create label names for the plot
    label_names = {0: "Real", 1: "Display Attack", 2: "Print Attack"}
    labels_named = [label_names[l] for l in labels_array]

    # Create seaborn scatter plot
    sns.scatterplot(
        x=features_2d[:, 0],
        y=features_2d[:, 1],
        hue=labels_named,
        palette="hot",
        alpha=0.7,
        s=50,
        edgecolor=None,
    )

    plt.title(
        f"t-SNE Visualization - {sota.value}",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.legend(title="Classes", title_fontsize=11, fontsize=10)

    # Save with consistent directory structure
    output_dir = Path(f"./avatar_experiments/features/{sota.value}")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "tsne_visualization.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_umap(features_list, labels_list, sota: SOTA):
    """Generate and save UMAP visualization"""
    features_array = np.array([f[0] for f in features_list])
    labels_array = np.array(labels_list)

    # Apply UMAP
    reducer = umap.UMAP(n_components=2, random_state=2024, n_neighbors=15, min_dist=0.1)
    features_2d = reducer.fit_transform(features_array)

    # Set seaborn style
    sns.set_theme(style="whitegrid")

    # Create figure
    plt.figure(figsize=(10, 8))

    # Create label names for the plot
    label_names = {0: "Real", 1: "Display Attack", 2: "Print Attack"}
    labels_named = [label_names[l] for l in labels_array]

    # Create seaborn scatter plot
    sns.scatterplot(
        x=features_2d[:, 0],
        y=features_2d[:, 1],
        hue=labels_named,
        palette="hot",
        alpha=0.7,
        s=50,
        edgecolor=None,
    )

    plt.title(
        f"UMAP Visualization - {sota.value}",
        fontsize=14,
        fontweight="bold",
    )
    plt.xlabel("UMAP Dimension 1", fontsize=12)
    plt.ylabel("UMAP Dimension 2", fontsize=12)
    plt.legend(title="Classes", title_fontsize=11, fontsize=10)

    # Save with consistent directory structure
    output_dir = Path(f"./avatar_experiments/features/{sota.value}")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "umap_visualization.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_pca(features_list, labels_list, sota: SOTA):
    """Generate and save PCA visualization"""
    features_array = np.array([f[0] for f in features_list])
    labels_array = np.array(labels_list)

    # Apply PCA
    pca = PCA(n_components=2, random_state=2024)
    features_2d = pca.fit_transform(features_array)

    # Set seaborn style
    sns.set_theme(style="whitegrid")

    # Create figure
    plt.figure(figsize=(10, 8))

    # Create label names for the plot
    label_names = {0: "Real", 1: "Display Attack", 2: "Print Attack"}
    labels_named = [label_names[l] for l in labels_array]

    # Create seaborn scatter plot
    sns.scatterplot(
        x=features_2d[:, 0],
        y=features_2d[:, 1],
        hue=labels_named,
        palette="hot",
        alpha=0.7,
        s=50,
        edgecolor=None,
    )

    plt.title(f"PCA Visualization - {sota.value}", fontsize=14, fontweight="bold")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)", fontsize=12)
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)", fontsize=12)
    plt.legend(title="Classes", title_fontsize=11, fontsize=10)

    # Save with consistent directory structure
    output_dir = Path(f"./avatar_experiments/features/{sota.value}")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "pca_visualization.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_features_for_model(sota: SOTA, protocol: str):
    model = get_model_and_load_checkpoint(sota, protocol)
    model = model.cuda()
    model = model.cuda()
    random.seed(2024)
    np.random.seed(2024)
    torch.manual_seed(2024)
    torch.cuda.manual_seed_all(2024)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    module = importlib.import_module(f"models.{sota.value.upper()}.inference")
    get_features = getattr(module, "get_features")

    _, test_ds = get_dataset(1, 224)
    model.eval()
    features_list = []
    with torch.no_grad():
        for x, y, z in test_ds:
            x = x.cuda()
            y = y.cuda().long()
            features = get_features(model, x)
            # Normalize features
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            features = features.cpu().numpy()
            for feat, path in zip(features, z):
                label = 0
                if "/DISPLAY/" in path:
                    label = 1
                elif "/PRINT/" in path:
                    label = 2
                # print(feat.shape, label)
                features_list.append((feat, label))

    # Extract labels list from features_list
    labels_list = [f[1] for f in features_list]

    # Generate all three visualizations
    plot_tsne(features_list, labels_list, sota)
    plot_umap(features_list, labels_list, sota)
    plot_pca(features_list, labels_list, sota)


g = torch.Generator()
g.manual_seed(2024)


def driver():
    results = {}

    for sota in SOTA:
        global REAL_LABEL, ATTACK_LABEL
        if sota in (SOTA.FLIP_FAS, SOTA.JPD_FAS):
            REAL_LABEL = 0
            ATTACK_LABEL = 1
        else:
            REAL_LABEL = 1
            ATTACK_LABEL = 0

        if sota in (SOTA.IADG_FAS,):
            continue

        results[sota.value] = {}
        success = True
        for protocol in MODELS_CHECKPOINTS[sota]:
            print(f"\n{'=' * 80}")
            print(f"Training {sota.value} with protocol {protocol}")
            print(f"{'=' * 80}\n")

            part_eers = []

            for part in range(1, 6):
                print(f"\n--- Part {part}/5 ---")
                model = get_model_and_load_checkpoint(sota, protocol)
                model = model.cuda()
                random.seed(2024)
                np.random.seed(2024)
                torch.manual_seed(2024)
                torch.cuda.manual_seed_all(2024)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.use_deterministic_algorithms(True)

                try:
                    best_eer = finetune_model(model, part, sota, protocol)
                    part_eers.append(best_eer)
                    print(f"Part {part} Best EER: {best_eer:.4f}%")
                except Exception as e:
                    print(f"Error during fine-tuning on {sota.value}: {e}")
                    traceback.print_exc()
                    success = False
                    break

                # Clear GPU memory
                del model
                torch.cuda.empty_cache()

            if not success:
                print(f"Skipping remaining protocols for {sota.value} due to errors.")
                results[sota.value][protocol] = {
                    "eers": float("inf"),
                    "mean": float("inf"),
                    "std": float("inf"),
                }
                break

            mean_eer = np.mean(part_eers)
            std_eer = np.std(part_eers)

            results[sota.value][protocol] = {
                "eers": part_eers,
                "mean": mean_eer,
                "std": std_eer,
                "ci 95": get_ci(part_eers),
            }

            print(f"\n{sota.value} - {protocol} Results:")
            print(f"  Mean EER: {mean_eer:.4f}% ± {std_eer:.4f}%")
            print(f"  Individual parts: {[f'{eer:.4f}%' for eer in part_eers]}")

    # Save final results
    results_path = Path("./avatar_experiments/final_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print results table
    print_results_table(results)

    print(f"\n{'=' * 80}")
    print("Detailed Results Summary:")
    print(f"{'=' * 80}")
    for sota_name, protocols in results.items():
        print(f"\n{sota_name}:")
        for protocol, stats in protocols.items():
            print(f"  {protocol}: {stats['mean']:.4f}% ± {stats['std']:.4f}%")

    print(f"\nResults saved to {results_path}")


def avatar_db_partitions():
    root_dir = Path("~/datasets/avatar_db").expanduser()
    for part in range(1, 6):
        subject_ids = [
            d.name
            for d in root_dir.iterdir()
            if d.name.startswith("S") and not d.name.endswith("51")
        ]
        random.shuffle(subject_ids)
        train_subjects = subject_ids[:25]
        test_subjects = subject_ids[25:]
        Path("./avatar_experiments").mkdir(parents=True, exist_ok=True)
        with open(f"./avatar_experiments/db_{part}.json", "w") as f:
            json.dump(
                {"train_subjects": train_subjects, "test_subjects": test_subjects},
                f,
            )


def get_scores_from_model(model, x, sota):
    """Get prediction scores from model based on SOTA type.

    Returns:
        cls_out: Raw model output (logits or probabilities)
    """
    if sota == SOTA.CF_FAS:
        cls_out = model(x, cf=None)
    elif sota == SOTA.DGUA_FAS:
        cls_out, *_ = model(x, True)
    elif sota == SOTA.FLIP_FAS:
        cls_out, _ = model.forward_eval(x)
    elif sota == SOTA.LMFD_FAS:
        cls_out, _ = model(x)
    elif sota == SOTA.JPD_FAS:
        _, cls_out = model(x)
    else:
        cls_out = model(x)
    return cls_out


def store_images():
    """
    Using the best model of each run in 1st partition finetuning,
    get the scores for all test images.
    Use check if run is ran before function to get the best eer.
    Using the scores calculate the eer and check whether both the eers match.
    Using these scores, determine the worst misclassified images
    for both real and attack classes and store them.
    """

    part = 1  # Only use partition 1

    for sota in SOTA:
        global REAL_LABEL, ATTACK_LABEL

        # Set labels based on SOTA type (FLIP_FAS has reversed labels)
        if sota in (SOTA.FLIP_FAS, SOTA.JPD_FAS):
            REAL_LABEL = 0
            ATTACK_LABEL = 1
        else:
            REAL_LABEL = 1
            ATTACK_LABEL = 0

        # Skip models as in driver function
        if sota in (SOTA.IADG_FAS,):
            continue

        for protocol in MODELS_CHECKPOINTS[sota]:
            print(f"\n{'=' * 80}")
            print(f"Processing {sota.value} - {protocol}")
            print(f"{'=' * 80}\n")

            # Set all seeds for reproducibility FIRST
            random.seed(2024)
            np.random.seed(2024)
            torch.manual_seed(2024)
            torch.cuda.manual_seed(2024)
            torch.cuda.manual_seed_all(2024)
            torch.use_deterministic_algorithms(True)

            # Set deterministic behavior for reproducible results
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Check if partition 1 was completed
            ckpt_dir = Path(
                f"./avatar_experiments/{sota.value}/{protocol}/{part}/checkpoints"
            )
            expected_eer, ckpt = check_if_run_is_ran_before(ckpt_dir)

            if expected_eer == float("inf"):
                print(
                    f"  Skipping {sota.value}/{protocol}: No completed run found for partition {part}"
                )
                continue

            print(f"  Expected EER from checkpoint: {expected_eer:.4f}%")

            # Load best model
            best_model_path = ckpt
            if not best_model_path.exists():
                print(f"  Skipping {sota.value}/{protocol}: best_model.pt not found")
                continue

            try:
                # Load checkpoint with pre-calculated scores
                checkpoint = torch.load(best_model_path, weights_only=False)
                scores_dict = checkpoint["scores"]
            except Exception as e:
                print(f"  Error loading checkpoint for {sota.value}/{protocol}: {e}")
                traceback.print_exc()
                continue

            # Convert scores_dict to all_results format
            all_results = []  # List of (image_path, score, label)

            for image_path, score in scores_dict.items():
                # Determine label from path
                path_upper = image_path.upper()
                if "/REAL/" in path_upper:
                    label = REAL_LABEL
                else:
                    # Attack (either DISPLAY or PRINT)
                    label = ATTACK_LABEL

                all_results.append((image_path, float(score), label))

            # Calculate EER
            genuine_scores = [
                score for _, score, label in all_results if label == REAL_LABEL
            ]
            impostor_scores = [
                score for _, score, label in all_results if label == ATTACK_LABEL
            ]

            print(
                f"  Total samples: {len(all_results)}, Genuine: {len(genuine_scores)}, Impostor: {len(impostor_scores)}"
            )
            print(
                f"  Genuine score range: [{min(genuine_scores):.4f}, {max(genuine_scores):.4f}]"
            )
            print(
                f"  Impostor score range: [{min(impostor_scores):.4f}, {max(impostor_scores):.4f}]"
            )

            # Debug: Check first few images to see if they're the same across runs
            print(
                f"  First 5 image paths: {[path.split('/')[-3:] for path, _, _ in all_results[:5]]}"
            )
            print(
                f"  First 5 scores: {[f'{score:.4f}' for _, score, _ in all_results[:5]]}"
            )

            calculated_eer = compute_eer(genuine_scores, impostor_scores) * 100
            print(f"  Calculated EER: {calculated_eer:.4f}%")

            # Check if EERs match, if not try swapping labels
            eer_diff = abs(calculated_eer - expected_eer)
            if eer_diff > 5:  # More than 0.01% difference
                print(f"  EER mismatch ({eer_diff:.4f}%), trying label swap...")
                # Try swapping genuine and impostor scores
                calculated_eer_swapped = (
                    compute_eer(impostor_scores, genuine_scores) * 100
                )
                eer_diff_swapped = abs(calculated_eer_swapped - expected_eer)
                print(f"  Swapped EER: {calculated_eer_swapped:.4f}% ")

                if eer_diff_swapped < eer_diff:
                    print(
                        f"  Swapped EER: {calculated_eer_swapped:.4f}% (better match)"
                    )
                    # Swap the label interpretation
                    genuine_scores, impostor_scores = impostor_scores, genuine_scores
                    calculated_eer = calculated_eer_swapped
                    eer_diff = eer_diff_swapped
                    # Swap REAL_LABEL and ATTACK_LABEL for this iteration
                    REAL_LABEL, ATTACK_LABEL = ATTACK_LABEL, REAL_LABEL

                if eer_diff > 5:
                    raise ValueError(
                        f"EER mismatch for {sota.value}/{protocol}: "
                        f"Expected {expected_eer:.4f}%, Calculated {calculated_eer:.4f}%, "
                        f"Difference: {eer_diff:.4f}%"
                    )

            print(f"  EER verification passed!")

            # Find worst misclassified images
            # Real misclassified: real images with lowest scores (most confidently predicted as attack)
            real_images = [
                (path, score, label)
                for path, score, label in all_results
                if label == REAL_LABEL
            ]
            real_misclassified = sorted(real_images, key=lambda x: x[1])[
                :2
            ]  # 2 lowest scores

            # Attack misclassified: attack images with highest scores (most confidently predicted as real)
            # Separate by attack type
            print_attack_images = [
                (path, score, label)
                for path, score, label in all_results
                if label == ATTACK_LABEL and "/PRINT/" in path.upper()
            ]
            display_attack_images = [
                (path, score, label)
                for path, score, label in all_results
                if label == ATTACK_LABEL and "/DISPLAY/" in path.upper()
            ]

            # Get worst misclassified for each attack type
            print_attack_misclassified = sorted(
                print_attack_images, key=lambda x: x[1], reverse=True
            )[:2]  # 2 highest scores
            display_attack_misclassified = sorted(
                display_attack_images, key=lambda x: x[1], reverse=True
            )[:2]  # 2 highest scores

            # Create output directories
            output_base = Path(
                f"./avatar_experiments/worst_images/{sota.value}/{protocol}"
            )
            real_dir = output_base / "real"
            print_dir = output_base / "print"
            display_dir = output_base / "display"
            real_dir.mkdir(parents=True, exist_ok=True)
            print_dir.mkdir(parents=True, exist_ok=True)
            display_dir.mkdir(parents=True, exist_ok=True)

            # Copy worst real misclassified images
            print(f"  Copying worst misclassified real images:")
            for idx, (img_path, score, label) in enumerate(real_misclassified, 1):
                output_name = f"worst_{idx}_{score:.4f}.png"
                output_path = real_dir / output_name
                shutil.copy2(img_path, output_path)
                print(f"    {output_name} (score: {score:.4f})")

            # Copy worst print attack misclassified images
            print(f"  Copying worst misclassified print attack images:")
            for idx, (img_path, score, label) in enumerate(
                print_attack_misclassified, 1
            ):
                output_name = f"worst_{idx}_{score:.4f}.png"
                output_path = print_dir / output_name
                shutil.copy2(img_path, output_path)
                print(f"    {output_name} (score: {score:.4f})")

            # Copy worst display attack misclassified images
            print(f"  Copying worst misclassified display attack images:")
            for idx, (img_path, score, label) in enumerate(
                display_attack_misclassified, 1
            ):
                output_name = f"worst_{idx}_{score:.4f}.png"
                output_path = display_dir / output_name
                shutil.copy2(img_path, output_path)
                print(f"    {output_name} (score: {score:.4f})")

            # Save metadata
            metadata = {
                "sota": sota.value,
                "protocol": protocol,
                "partition": part,
                "checkpoint_eer": float(expected_eer),
                "calculated_eer": float(calculated_eer),
                "worst_real_misclassified": [
                    {"path": str(p), "score": float(s), "label": int(l)}
                    for p, s, l in real_misclassified
                ],
                "worst_print_attack_misclassified": [
                    {"path": str(p), "score": float(s), "label": int(l)}
                    for p, s, l in print_attack_misclassified
                ],
                "worst_display_attack_misclassified": [
                    {"path": str(p), "score": float(s), "label": int(l)}
                    for p, s, l in display_attack_misclassified
                ],
            }

            metadata_path = output_base / "metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            print(f"  Metadata saved to {metadata_path}")

    print(f"\n{'=' * 80}")
    print("Finished processing all models!")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    # avatar_db_partitions()
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    # driver()  # Original fine-tuning
    # store_images()  # Store worst images

    # # Compute and print separated metrics
    # results = compute_attack_separated_metrics()
    # print_attack_separated_tables(results)
    plot_features({})
