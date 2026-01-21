import json

import numpy as np
from PIL import ImageFile
from sklearn.metrics import auc, roc_curve

from util import SOTA

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Constants for protocols and datasets
PROTOCOLS = ["icmo", "oimc", "ocim", "ocmi"]
DATASETS = ["Oulu", "Casia_fasd", "Msu_mfsd", "Idiap", "IphoneIndia", "Rose_youtu"]


def get_err_threhold(fpr, tpr, threshold):
    differ_tpr_fpr_1 = tpr + fpr - 1.0
    right_index = np.argmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    err = fpr[right_index]
    return err, best_th, right_index


def performances_val(val_scores, val_labels, fpr_rate=0.05):
    data = []
    count = 0.0
    num_real = 0.0
    num_fake = 0.0
    for score, label in zip(val_scores, val_labels):
        data.append({"map_score": score, "label": label})
        count += 1
        if label == 1:
            num_real += 1
        else:
            num_fake += 1

    fpr, tpr, threshold = roc_curve(
        val_labels, val_scores, pos_label=1, drop_intermediate=False
    )
    # print(threshold.shape, fpr.shape, tpr.shape)
    # print(threshold.min(), threshold.max())

    auc_test = auc(fpr, tpr)

    tpr_at_fpr = tpr[np.argmin(abs(fpr - fpr_rate))]

    val_err, val_threshold, right_index = get_err_threhold(fpr, tpr, threshold)
    # print("threshold:", val_threshold)

    type1 = len([s for s in data if s["map_score"] < val_threshold and s["label"] == 1])
    type2 = len([s for s in data if s["map_score"] > val_threshold and s["label"] == 0])

    val_ACC = 1 - (type1 + type2) / count

    FRR = 1 - tpr  # FRR = 1 - TPR

    HTER = (fpr + FRR) / 2.0  # error recognition rate &  reject recognition rate

    return (
        val_ACC,
        fpr[right_index],
        FRR[right_index],
        HTER[right_index],
        auc_test,
        val_err,
        tpr_at_fpr,
        val_threshold,
    )


def compute_eer(real_scores, attack_scores, reverse=False):
    real_scores = np.array([float(score[1]) for score in real_scores])
    attack_scores = np.array([float(score[1]) for score in attack_scores])
    if reverse:
        real_scores = 1 - real_scores
        attack_scores = 1 - attack_scores

    thresholds = np.linspace(
        min(real_scores.min(), attack_scores.min()),
        max(real_scores.max(), attack_scores.max()),
        10_001,
    )

    fars = np.zeros_like(thresholds)
    frrs = np.zeros_like(thresholds)
    # print(real_scores.shape, attack_scores.shape)
    # print(real_scores[:5])
    # print(real_scores.shape[0], np.sum(real_scores <= 1))
    for i, threshold in np.ndenumerate(thresholds):
        frr = np.sum(real_scores <= threshold) / real_scores.shape[0]
        far = np.sum(attack_scores > threshold) / attack_scores.shape[0]
        fars[i] = far
        frrs[i] = frr

    # idx = np.argmin(np.abs(fars - 0.05))
    idx = np.argmin(np.abs(fars - frrs))
    eer = (fars[idx] + frrs[idx]) / 2
    # print("EEr threshold:", thresholds[np.argmin(np.abs(fars - frrs))])
    return (
        eer * 100,
        auc(fars, 1 - frrs) * 100,
        thresholds[idx],
        (1 - frrs[np.argmin(np.abs(fars - 0.01))]) * 100,  # TPR at FPR=1%
        (1 - frrs[np.argmin(np.abs(fars - 0.05))]) * 100,  # TPR at FPR=5%
    )


def load_and_compute_eer(sota, protocol, dataset):
    """
    Load result files and compute EER for a specific combination.

    Args:
        sota: SOTA method name (lowercase)
        protocol: Training protocol name
        dataset: Test dataset name

    Returns:
        float: EER percentage, or None if files don't exist
    """
    real_fname = f"./results/{sota}/{protocol}/{dataset}/real.json"
    attack_fname = f"./results/{sota}/{protocol}/{dataset}/attack.json"

    try:
        with open(real_fname) as f:
            real_data = json.load(f)
        with open(attack_fname) as f:
            attack_data = json.load(f)

        real_scores = real_data["result"]
        attack_scores = attack_data["result"]

        # Special case: GACD_FAS + oimc requires reversed scores
        reverse = protocol == "oimc" and sota == SOTA.GACD_FAS.value.lower()

        eer, _, _, _, _ = compute_eer(real_scores, attack_scores, reverse)
        return eer

    except FileNotFoundError:
        return None


def format_matrix_row(dataset_name, eer_values):
    """
    Format a single row of the matrix.

    Args:
        dataset_name: Name of the dataset (row label)
        eer_values: List of 4 EER values (one per protocol), None for missing

    Returns:
        str: Formatted row string
    """
    # Column widths: dataset=15, each protocol=10
    row = f"{dataset_name:<15}"

    for eer in eer_values:
        if eer is None:
            row += f"{'   -':<10}"
        else:
            row += f"{eer:>6.2f}%   "

    return row


def print_model_matrix(sota, protocols, datasets):
    """
    Print the complete matrix for one SOTA model.

    Args:
        sota: SOTA method name (lowercase)
        protocols: List of protocol names (4 protocols)
        datasets: List of dataset names (5 datasets)
    """
    # Print header
    print(f"\nSOTA Method: {sota.upper()}")
    print("=" * 70)

    # Print column headers
    header = f"{'Dataset':<15}"
    for protocol in protocols:
        header += f"{protocol.upper():<10}"
    print(header)
    print("-" * 70)

    # Print each dataset row
    for dataset in datasets:
        eer_values = []
        for protocol in protocols:
            if (
                protocol != "all"
                and protocol[-1].lower() != dataset[0].lower()
                and dataset not in ["IphoneIndia", "Rose_youtu"]
            ):
                eer_values.append(-1)
                continue
            eer = load_and_compute_eer(sota, protocol, dataset)
            eer_values.append(eer)

        print(format_matrix_row(dataset, eer_values))

    print()


sotas = [
    SOTA.CF_FAS,
    SOTA.DGUA_FAS,
    SOTA.LMFD_FAS,
    SOTA.FLIP_FAS,
    SOTA.JPD_FAS,
    SOTA.IADG_FAS,
    SOTA.GACD_FAS,
]
for sota_enum in sotas:
    sota = sota_enum.value.lower()

    # Special case for JPD_FAS
    if sota == SOTA.JPD_FAS.value.lower():
        protocols = ["all", "all", "all", "all"]
    else:
        protocols = PROTOCOLS

    print_model_matrix(sota, protocols, DATASETS)
