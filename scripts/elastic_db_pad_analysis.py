import json
import os
import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from eval_loop import MODELS_CHECKPOINTS, SOTA


def compute_eer(bonafide_scores, attack_scores):
    bonafide_scores = np.array(bonafide_scores)
    attack_scores = np.array(attack_scores)
    thresholds = np.linspace(0, 1, 10_001)
    apcers, bpcers = np.ones_like(thresholds), np.ones_like(thresholds)
    for i, threshold in enumerate(thresholds):
        bpcers[i] = np.sum(bonafide_scores <= threshold) / len(bonafide_scores)
        apcers[i] = np.sum(attack_scores > threshold) / len(attack_scores)
    apcers *= 100
    bpcers *= 100
    eer_index = np.argmin(np.abs(apcers - bpcers))
    eer = (apcers[eer_index] + bpcers[eer_index]) / 2
    threshold = thresholds[eer_index]
    return eer, threshold, apcers, bpcers


def call_proc(call: str) -> None:
    print(call)
    subprocess.run(
        call,
        shell=True,
        executable="/bin/bash",
        #         capture_output=True,
        #         text=True,
    )


def get_bonafide_paths():
    rdir = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/Elastic_3D_Mask_FD/Elastic_3D_Mask_FD/PAD_DB_Structured/Final_Bona/iPhone11"
    bonafide_paths = []
    for p in Path(rdir).rglob("*"):
        if p.is_file() and p.suffix.lower() in [".jpg", ".png", ".jpeg"]:
            bonafide_paths.append(str(p))

    return bonafide_paths


def get_attack_paths():
    rdir = "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/Elastic_3D_Mask_FD/Elastic_3D_Mask_FD/PAD_DB_Structured/Attack/iPhone11/Human_Back/Face_Detect"
    attack_paths = []
    for p in Path(rdir).rglob("*"):
        if p.is_file() and p.suffix.lower() in [".jpg", ".png", ".jpeg"]:
            attack_paths.append(str(p))
    return attack_paths


def retrieve_result(
    files: list[str],
    container_name: str,
    ckpt: str,
    temp_fname: str = "tmp.json",
) -> list[float]:
    temp_input = "input_" + temp_fname
    temp_output = "output_" + temp_fname

    with open(f"./models/{container_name.upper()}/{temp_input}", "w+") as fp:
        json.dump({"files": files}, fp)

    call_proc(
        f"docker exec {container_name} python /root/code/inference.py -i {temp_input} -o {temp_output} -ckpt '{ckpt}'"
    )

    with open(f"./models/{container_name.upper()}/{temp_output}", "r") as fp:
        output = json.load(fp)

    os.remove(f"./models/{container_name.upper()}/{temp_input}")
    os.remove(f"./models/{container_name.upper()}/{temp_output}")
    return output["result"]


def get_results_for_sota(sota: SOTA, data: dict[str, list[str]]):
    for data_name, paths in data.items():
        for protocol, ckpt in MODELS_CHECKPOINTS[sota].items():
            save_path = Path(f"result/{sota.name.lower()}/{protocol}/{data_name}.npy")
            save_path.parent.mkdir(parents=True, exist_ok=True)
            if save_path.exists():
                print(
                    f"Results already exist for {sota.name} on {data_name} with {protocol}. Skipping."
                )
                continue

            scores = retrieve_result(paths, sota.name.lower(), ckpt)
            np.save(save_path, scores)


def get_file_paths() -> dict[str, list[str]]:
    tmp_fpath = Path("./scripts/filenames.json")
    if tmp_fpath.exists():
        with open(tmp_fpath, "r") as f:
            return json.load(f)
    data = {}
    data["iPhone11_bonafide"] = []
    for p in tqdm(
        Path(
            "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/Elastic_3D_Mask_FD/Elastic_3D_Mask_FD/PAD_DB_Structured/Final_Bona/iPhone11"
        ).glob("*"),
        desc="Copying iPhone11 bonafide files",
    ):
        if p.is_file() and p.suffix.lower() in [".jpg", ".png", ".jpeg"]:
            output_path = Path(
                str(p).replace(
                    "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu",
                    "/home/ubuntu/datasets",
                )
            )
            if not output_path.exists():
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, output_path)
            data["iPhone11_bonafide"].append(
                str(output_path).replace("/home/ubuntu", "/root")
            )
    data["iPhone12_bonafide"] = []
    for p in tqdm(
        Path(
            "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/3D_PAD_Datasets/2D_Face_Databases_PAD/iPhone12/Data_Split/real/train/Face_Detect"
        ).glob("*"),
        desc="Copying iPhone12 bonafide files",
    ):
        if p.is_file() and p.suffix.lower() in [".jpg", ".png", ".jpeg"]:
            output_path = Path(
                str(p).replace(
                    "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu",
                    "/home/ubuntu/datasets",
                )
            )
            if not output_path.exists():
                output_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, output_path)
            data["iPhone12_bonafide"].append(
                str(output_path).replace("/home/ubuntu", "/root")
            )

    rpath = Path(
        "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/Elastic_3D_Mask_FD/Elastic_3D_Mask_FD/PAD_DB_Structured/Attack/"
    )
    for iphone in ["iPhone11", "iPhone12"]:
        for attack in ["Human_Back", "Human_Front", "Manquen_Back", "Manquen_Front"]:
            data[f"{iphone}_{attack}"] = []
            for p in tqdm(
                (rpath / iphone / attack / "Face_Detect2").glob("*"),
                desc=f"Copying {iphone} {attack} files",
            ):
                if p.is_file() and p.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                    output_path = Path(
                        str(p).replace(
                            "/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu",
                            "/home/ubuntu/datasets",
                        )
                    )
                    if not output_path.exists():
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(p, output_path)

                    data[f"{iphone}_{attack}"].append(
                        str(output_path).replace("/home/ubuntu", "/root")
                    )

    with open(tmp_fpath, "w+") as f:
        json.dump(data, f)
    return data


def driver() -> None:
    print("Getting file paths")
    data = get_file_paths()
    print("Getting results for SOTA")
    for sota in SOTA:
        print(f"Getting results for {sota.name}")
        get_results_for_sota(sota, data)


def print_results_table(
    results: dict, metric_name: str, metric_suffix: str = ""
) -> None:
    """
    Print a formatted table of results.

    Args:
        results: Dictionary of model results in format {model_name: {key: (mean, std), ...}}
        metric_name: Name of the metric to display in table title (e.g., "EER", "BPCER@5%APCER")
        metric_suffix: Suffix to append to the data keys when looking up results (e.g., "_bpcers_at_5_apcer")
    """
    attacks = ["Human_Back", "Human_Front", "Manquen_Back", "Manquen_Front"]
    iphones = ["iPhone11", "iPhone12"]

    # Create and print the table
    print(f"\n{metric_name} Results Table:")

    # Calculate column widths
    col_width = 14  # Width for each attack column (increased for full names)
    model_col_width = 10  # Width for model name column

    # Calculate total table width
    total_width = (
        model_col_width + 2 + (len(iphones) * len(attacks) * (col_width + 1)) + 1
    )
    print("=" * total_width)

    # Create header
    header = f"| {'':^{model_col_width}} |"
    for iphone in iphones:
        # Calculate the width needed for this iPhone section
        iphone_width = len(attacks) * (col_width + 1) - 1
        header += f" {iphone:^{iphone_width}} |"
    print(header)

    # Create subheader with attack names
    subheader = f"| {'Model':^{model_col_width}} |"
    for iphone in iphones:
        for attack in attacks:
            # Use cleaner attack names that fit in the column width
            attack_display = attack.replace("_", " ")  # Replace underscore with space
            subheader += f" {attack_display:^{col_width}} |"
    print(subheader)

    # Print separator
    separator = "|" + "-" * (model_col_width + 1) + "|"
    for iphone in iphones:
        for attack in attacks:
            separator += "-" * (col_width + 1) + "|"
    print(separator)

    # Print results for each model
    for model_name, model_results in results.items():
        row = f"| {model_name:<{model_col_width}} |"
        for iphone in iphones:
            for attack in attacks:
                key = f"{iphone}_{attack}{metric_suffix}"
                if key in model_results:
                    mean_val, std_val = model_results[key]
                    if std_val > 0:
                        cell = f"{mean_val:.3f}±{std_val:.3f}"
                    else:
                        cell = f"{mean_val:.3f}"
                    row += f" {cell:^{col_width}} |"
                else:
                    row += f" {'N/A':^{col_width}} |"
        print(row)

    print("=" * total_width)


def get_eer_tables() -> None:
    """
    Compute eer and print a table in following format:
    |       |    Iphone 11                |    Iphone 12          |
    | Model | Attack1 | Attack2 | Attack3 | Attack1 | Attack2 | Attack3 |
    |-------|--------|-----------|--------|-----------|--------|-----------|
    | SOTA1 | 0.01   | 0.02 ±0.01 | 0.03 ±0.01 | 0.04 ±0.01 | 0.05 ±0.01 | 0.06 ±0.01 |
    | SOTA2 | 0.02   | 0.03 ±0.01 | 0.04 ±0.01 | 0.05 ±0.01 | 0.06 ±0.01 | 0.07 ±0.01 |
    | SOTA3 | 0.03   | 0.04 ±0.01 | 0.05 ±0.01 | 0.06 ±0.01 | 0.07 ±0.01 | 0.08 ±0.01 |
    | SOTA4 | 0.04   | 0.05 ±0.01 | 0.06 ±0.01 | 0.07 ±0.01 | 0.08 ±0.01 | 0.09 ±0.01 |

    The Attacks are:
    - Human_Back
    - Human_Front
    - Manquen_Back
    - Manquen_Front
    The Iphones are:
    - iPhone11
    - iPhone12
    Except JPD_FAS from SOTA, check whether scores of all the protocols are present and then
    compute eer for each protocol and show the mean and std of eer for each model.
    Skip JPD_FAS from SOTA.
    """
    # Define attacks and iPhones
    attacks = ["Human_Back", "Human_Front", "Manquen_Back", "Manquen_Front"]
    iphones = ["iPhone11", "iPhone12"]

    # Skip JPD_FAS as requested
    sota_models = [sota for sota in SOTA if sota != SOTA.JPD_FAS]

    results = {}

    for sota in sota_models:
        model_name = sota.name
        sota_dir = Path(f"./result/{sota.name.lower()}")

        if not sota_dir.exists():
            print(f"Warning: No results found for {model_name}")
            continue

        model_results = {}

        # Get all protocols for this model
        protocols = list(MODELS_CHECKPOINTS[sota].keys())

        for iphone in iphones:
            for attack in attacks:
                eers_for_attack = []
                bpcers_at_5_apcer_for_attack = []
                bpcers_at_10_apcer_for_attack = []

                # Check if all protocols have results for this attack
                protocols_with_results = []
                for protocol in protocols:
                    bonafide_path = sota_dir / protocol / f"{iphone}_bonafide.npy"
                    attack_path = sota_dir / protocol / f"{iphone}_{attack}.npy"

                    if bonafide_path.exists() and attack_path.exists():
                        protocols_with_results.append(protocol)

                if not protocols_with_results:
                    print(
                        f"Warning: No complete results for {model_name} {iphone} {attack}"
                    )
                    continue

                # Compute EER for each protocol that has results
                for protocol in protocols_with_results:
                    bonafide_path = sota_dir / protocol / f"{iphone}_bonafide.npy"
                    attack_path = sota_dir / protocol / f"{iphone}_{attack}.npy"

                    bonafide_scores = [
                        float(x) for x in np.load(bonafide_path)[:, 1].tolist()
                    ]
                    attack_scores = [
                        float(x) for x in np.load(attack_path)[:, 1].tolist()
                    ]

                    eer, _, apcers, bpcers = compute_eer(bonafide_scores, attack_scores)
                    bpcers_at_5_apcer_for_attack.append(
                        bpcers[np.argmin(np.abs(apcers - 5))]
                    )
                    bpcers_at_10_apcer_for_attack.append(
                        bpcers[np.argmin(np.abs(apcers - 10))]
                    )
                    eers_for_attack.append(eer)

                if eers_for_attack:
                    mean_eer = np.mean(eers_for_attack)
                    std_eer = (
                        np.std(eers_for_attack) if len(eers_for_attack) > 1 else 0.0
                    )
                    model_results[f"{iphone}_{attack}"] = (mean_eer, std_eer)
                    model_results[f"{iphone}_{attack}_bpcers_at_5_apcer"] = (
                        np.mean(bpcers_at_5_apcer_for_attack),
                        np.std(bpcers_at_5_apcer_for_attack),
                    )
                    model_results[f"{iphone}_{attack}_bpcers_at_10_apcer"] = (
                        np.mean(bpcers_at_10_apcer_for_attack),
                        np.std(bpcers_at_10_apcer_for_attack),
                    )

        if model_results:
            results[model_name] = model_results

    # Print EER table
    print_results_table(results, "EER")

    # Print BPCER@5%APCER table
    print_results_table(results, "BPCER@5%APCER", "_bpcers_at_5_apcer")

    # Print BPCER@10%APCER table
    print_results_table(results, "BPCER@10%APCER", "_bpcers_at_10_apcer")


def plot_apcer_bpcer_curves_by_attack(save_dir: str = "./plots") -> None:
    """
    Plot APCER vs BPCER curves for each attack type separately.
    For each attack, create one figure showing all available SOTA models.
    For each model, show a light faded range for all protocols and a bold mean curve.
    Uses seaborn for plotting.

    Args:
        save_dir: Directory to save the plots
    """
    # Define attacks and iPhones
    attacks = ["Human_Back", "Human_Front", "Manquen_Back", "Manquen_Front"]
    iphones = ["iPhone11", "iPhone12"]

    # Skip JPD_FAS as requested
    sota_models = [sota for sota in SOTA if sota != SOTA.JPD_FAS]

    # Create save directory if it doesn't exist
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Set up the plot style
    sns.set_style("whitegrid")

    # Color palette for different models
    colors = sns.color_palette("tab10", len(sota_models))

    # Create a separate plot for each attack
    for iphone in iphones:
        for attack in attacks:
            plt.figure(figsize=(12, 8))

            for model_idx, sota in enumerate(sota_models):
                if sota in [SOTA.JPD_FAS, SOTA.IADG_FAS]:
                    continue
                model_name = sota.name
                sota_dir = Path(f"./result/{sota.name.lower()}")

                if not sota_dir.exists():
                    print(f"Warning: No results found for {model_name}")
                    continue

                # Get all protocols for this model
                protocols = list(MODELS_CHECKPOINTS[sota].keys())

                # Collect all APCER/BPCER curves for this model and attack
                all_apcers = []
                all_bpcers = []

                protocols_with_results = []

                # Check which protocols have complete results
                for protocol in protocols:
                    bonafide_path = sota_dir / protocol / f"{iphone}_bonafide.npy"
                    attack_path = sota_dir / protocol / f"{iphone}_{attack}.npy"

                    if bonafide_path.exists() and attack_path.exists():
                        protocols_with_results.append(protocol)

                # Compute curves for each protocol
                for protocol in protocols_with_results:
                    bonafide_path = sota_dir / protocol / f"{iphone}_bonafide.npy"
                    attack_path = sota_dir / protocol / f"{iphone}_{attack}.npy"

                    bonafide_scores = [
                        float(x) for x in np.load(bonafide_path)[:, 1].tolist()
                    ]
                    attack_scores = [
                        float(x) for x in np.load(attack_path)[:, 1].tolist()
                    ]

                    eer, threshold, apcers, bpcers = compute_eer(
                        bonafide_scores, attack_scores
                    )
                    modified_bpcers = np.ones_like(bpcers) * 100
                    modified_apcers = np.linspace(0, 100, 10_001)
                    for i, apcer in enumerate(modified_apcers):
                        modified_bpcers[i] = bpcers[np.argmin(np.abs(apcers - apcer))]

                    all_apcers.append(modified_apcers)
                    all_bpcers.append(modified_bpcers)

                if not all_apcers:
                    print(f"Warning: No data found for {model_name} on {attack}")
                    continue

                # Convert to numpy arrays for easier manipulation
                all_apcers = np.array(all_apcers)
                all_bpcers = np.array(all_bpcers)

                # Compute mean and std for confidence intervals
                mean_bpcers = np.mean(all_bpcers, axis=0)
                std_bpcers = np.std(all_bpcers, axis=0)

                # Create data for seaborn lineplot
                plot_data = pd.DataFrame(
                    {
                        "APCER": all_apcers[
                            0
                        ],  # Since all_bpcers are the same (modified_bpcers)
                        "BPCER_mean": mean_bpcers,
                        "BPCER_lower": mean_bpcers - std_bpcers,
                        "BPCER_upper": mean_bpcers + std_bpcers,
                    }
                )

                # Plot mean line with seaborn
                sns.lineplot(
                    data=plot_data,
                    x="APCER",
                    y="BPCER_mean",
                    color=colors[model_idx],
                    linewidth=3,
                    label=f"{model_name.split('_')[0]}",
                )

                # Add confidence interval
                plt.fill_between(
                    plot_data["APCER"],
                    plot_data["BPCER_lower"],
                    plot_data["BPCER_upper"],
                    color=colors[model_idx],
                    alpha=0.2,
                )

            # Plot diagonal line for EER reference
            plt.plot([0, 80], [0, 80], "k--", alpha=0.5, linewidth=1, label="EER line")

            # Customize the plot
            plt.xlabel(
                "APCER (Attack Presentation Classification Error Rate) %", fontsize=16
            )
            plt.ylabel(
                "BPCER (Bonafide Presentation Classification Error Rate) %", fontsize=16
            )
            # plt.title(
            #     f"APCER vs BPCER Curves - {attack} Attack - {iphone}",
            #     fontsize=18,
            #     fontweight="bold",
            # )
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=16)
            plt.grid(True, alpha=0.3)
            plt.xlim(0, 80)
            plt.ylim(0, 80)

            # Make the plot tight and save
            plt.tight_layout()
            save_path = Path(save_dir) / f"{iphone}_{attack.lower()}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.show()

            print(f"APCER vs BPCER plot for {attack} saved to {save_path}")


def get_bets_image():
    attacks = ["Human_Back", "Human_Front", "Manquen_Back", "Manquen_Front"]
    iphones = ["iPhone11", "iPhone12"]

    # Skip JPD_FAS as requested
    sota_models = [sota for sota in SOTA if sota != SOTA.JPD_FAS]

    # Create a separate plot for each attack
    for iphone in iphones:
        for attack in attacks:
            plt.figure(figsize=(12, 8))

            for model_idx, sota in enumerate(sota_models):
                if sota in [SOTA.JPD_FAS, SOTA.IADG_FAS]:
                    continue
                model_name = sota.name
                sota_dir = Path(f"./result/{sota.name.lower()}")

                if not sota_dir.exists():
                    print(f"Warning: No results found for {model_name}")
                    continue

                # Get all protocols for this model
                protocols = list(MODELS_CHECKPOINTS[sota].keys())

                # Compute curves for each protocol
                for protocol in protocols:
                    bonafide_path = sota_dir / protocol / f"{iphone}_bonafide.npy"
                    attack_path = sota_dir / protocol / f"{iphone}_{attack}.npy"
                    sorted_bonafide = sorted(
                        np.load(bonafide_path).tolist(), key=lambda x: float(x[1])
                    )
                    sorted_attack = sorted(
                        np.load(attack_path).tolist(), key=lambda x: float(x[1])
                    )
                    max_case_bona = sorted_bonafide[-1]
                    max_case_attack = sorted_attack[-1]
                    min_case_bona = sorted_bonafide[0]
                    min_case_attack = sorted_attack[0]
                    os.makedirs(
                        f"./plots/{iphone}_{attack}/{model_name}/{protocol}",
                        exist_ok=True,
                    )
                    shutil.copy(
                        max_case_bona[0].replace("/root", "/home/ubuntu"),
                        f"./plots/{iphone}_{attack}/{model_name}/{protocol}/max_case_bona.png",
                    )
                    shutil.copy(
                        max_case_attack[0].replace("/root", "/home/ubuntu"),
                        f"./plots/{iphone}_{attack}/{model_name}/{protocol}/max_case_attack.png",
                    )
                    shutil.copy(
                        min_case_bona[0].replace("/root", "/home/ubuntu"),
                        f"./plots/{iphone}_{attack}/{model_name}/{protocol}/min_case_bona.png",
                    )
                    shutil.copy(
                        min_case_attack[0].replace("/root", "/home/ubuntu"),
                        f"./plots/{iphone}_{attack}/{model_name}/{protocol}/min_case_attack.png",
                    )


if __name__ == "__main__":
    # driver()
    # get_eer_tables()
    # plot_apcer_bpcer_curves_by_attack()
    get_bets_image()
