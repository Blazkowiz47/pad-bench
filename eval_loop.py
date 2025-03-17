import json
import os
from os.path import join as pjoin
import re
import subprocess
from glob import glob
from typing import Dict, List
from matplotlib import pyplot as plt
from util import SOTA, image_extensions
import numpy as np

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
        "icmo": "/root/pretrained_models/DGUA_FAS/I&C&MtoO/best_model.pth.tar",
        "oimc": "/root/pretrained_models/DGUA_FAS/O&I&MtoC/best_model.pth.tar",
        "ocim": "/root/pretrained_models/DGUA_FAS/O&C&ItoM/best_model.pth.tar",
        "ocmi": "/root/pretrained_models/DGUA_FAS/O&C&MtoI/best_model.pth.tar",
    },
    SOTA.FLIP_FAS: {
        "icmo": "/root/pretrained_models/FLIP_FAS/oulu_flip_mcl.pth.tar",
        "oimc": "/root/pretrained_models/FLIP_FAS/casia_flip_mcl.pth.tar",
        "ocim": "/root/pretrained_models/FLIP_FAS/msu_flip_mcl.pth.tar",
        "ocmi": "/root/pretrained_models/FLIP_FAS/replay_flip_mcl.pth.tar",
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


def check_dir(dir) -> bool:
    flag = True
    for ssplit in ["test", "train"]:
        for ty in ["real", "attack"]:
            flag = flag and os.path.isdir(os.path.join(dir, f"{ssplit}_{ty}.txt"))
    return flag


def get_checkpoint_path(sota: SOTA, iphone: str, attack: str) -> str:
    if sota == SOTA.JPD_FAS:
        models = glob(f"./tmp/JPD_FAS/{iphone}/{attack}/checkpoints/resnet50_*.pth")
        best_accuracy = 0
        best_model = None
        for model in models:
            pattern = r"_acc1_([0-9]+\.[0-9]+)\.pth$"
            match = re.search(pattern, model)
            if match:
                accuracy = float(match.group(1))
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = model
        if not best_model:
            raise ValueError(f"JPD_FAS not trained for {iphone} and {attack}")
        return best_model

    if sota == SOTA.LMFD_FAS:
        return f"./tmp/LMFD_FAS/{iphone}/{attack}/checkpoints/best_weights.pth"

    if sota == SOTA.CF_FAS:
        train_log = f"./tmp/CF_FAS/{iphone}/{attack}/train.txt"
        with open(train_log, "r") as fp:
            lines = fp.readlines()
        pattern = r"Epoch:\s*(\d+),.*loss_total=\s*([0-9]*\.[0-9]+)"
        min_loss = float("inf")
        best_model = None

        for line in lines:
            match = re.search(pattern, line)
            if match:
                epoch_number = match.group(1)
                loss_total_value = float(match.group(2))
                if min_loss > loss_total_value:
                    best_model = (
                        f"./tmp/CF_FAS/{iphone}/{attack}/checkpoints/{epoch_number}.pth"
                    )
                    min_loss = loss_total_value

        if not best_model:
            raise ValueError(f"CF_FAS not trained for {iphone} and {attack}")

        return best_model

    if sota == SOTA.IADG_FAS:
        return f"./tmp/IADG_FAS/{iphone}/{attack}/model_best.pth.tar"

    raise ValueError(f"SOTA: {sota} not trained for {iphone} and {attack}")


def call_proc(call: str) -> None:
    subprocess.run(
        call,
        shell=True,
        executable="/bin/bash",
        #         capture_output=True,
        #         text=True,
    )


def get_files(rdir: str) -> List[str]:
    result_files = []
    for root, _, files in os.walk(rdir):
        for file in files:
            if "." + file.split(".")[-1].lower() in image_extensions:
                fname = os.path.join(root, file)
                fname = fname.replace("/home/ubuntu", "/root")
                result_files.append(fname)

    return result_files


def get_files_for_idiap() -> Dict[str, List[str]]:
    rdir = "/home/ubuntu/datasets/test/idiap/"
    results: Dict[str, List[str]] = {}
    for version in ["v1", "v2"]:
        dir = pjoin(rdir, "real", version)
        results["real_" + version] = get_files(dir)

        for attack in ["display", "print"]:
            dir = pjoin(rdir, "attack", attack, version)
            results[attack + "_" + version] = get_files(dir)

    return results


def get_files_for_nexdata() -> Dict[str, List[str]]:
    rdir = "/home/ubuntu/datasets/test/nexdata"
    results: Dict[str, List[str]] = {}
    results["real"] = get_files(pjoin(rdir, "real"))
    for attack in [
        "display_ipad",
        "display_mobilephone",
        "print_paper_mask",
        "print_paper_photo",
    ]:
        dir = pjoin(rdir, "attack", attack)
        results[attack] = get_files(dir)

    return results


def get_files_for_hda() -> Dict[str, List[str]]:
    rdir = "/home/ubuntu/datasets/test/hda_flicker/attack/"
    results: Dict[str, List[str]] = {}
    results["real"] = get_files("/home/ubuntu/datasets/test/hda_flicker/real")
    for attack in os.listdir(rdir):
        dir = pjoin(rdir, attack)
        if os.path.isdir(dir):
            results[attack] = get_files(dir)

    return results


def get_files_for_trainingpro2() -> Dict[str, List[str]]:
    rdir = "/home/ubuntu/datasets/test/trainingProiBeta2"
    results: Dict[str, List[str]] = {}
    results["real"] = get_files(pjoin(rdir, "real"))
    for attack in ["latex", "silicon", "wrap"]:
        dir = pjoin(rdir, "attack", attack)
        results[attack] = get_files(dir)

    return results


def retrieve_result(
    files: List[str],
    container_name: str,
    ckpt: str,
    temp_fname: str = "tmp_input.json",
) -> List[float]:
    temp_input = "input_" + temp_fname
    temp_output = "output_" + temp_fname

    with open(pjoin(f"./models/{container_name.upper()}/{temp_input}"), "w+") as fp:
        json.dump({"files": files}, fp)

    call_proc(
        f"docker exec {container_name} python /root/code/inference.py -i {temp_input} -o {temp_output} -ckpt {ckpt}"
    )

    with open(pjoin(f"./models/{container_name.upper()}/{temp_output}"), "r") as fp:
        output = json.load(fp)
    os.remove(pjoin(f"./models/{container_name.upper()}/{temp_input}"))
    os.remove(pjoin(f"./models/{container_name.upper()}/{temp_output}"))
    return output["result"]


def eval_all_indian_pads() -> None:
    for sota in MODELS_CHECKPOINTS:
        for protocol, ckpt in MODELS_CHECKPOINTS[sota].items():
            for dname in [
                "3D_mask_google",
                #         "HDA_flicker_attacks",
                "idiap",
                "test_dataset_pro",
                "training_pro_ibeta_level2",
            ]:
                #                 sota = SOTA.DGUA_FAS
                model = sota.name
                rdir = "/home/ubuntu/datasets/test/"
                edir = os.path.join("./tmp/", model, protocol, dname)
                #             ckpt = (
                #                 "./models/DGUA_FAS/test_checkpoint/dgua_fas/best_model/best_model.pth.tar"
                #             )

                source_call = "source ~/miniconda3/etc/profile.d/conda.sh"

                syscall = f'python evaluation.py -m {model} \
                        -c "configs/train.yaml" -d standard \
                        --attack="*" --rdir={rdir} \
                        --batch-size={BATCH_SIZE} -ckpt "{ckpt}" \
                        --dataset-name={dname} \
                        -edir {edir}  --logger-level=ERROR'
                conda_env = model.lower().replace("_", "")
                print("Protocol:", protocol, "model:", model)
                subprocess.run(
                    f"{source_call}; conda activate {conda_env}; {syscall}; conda deactivate",
                    shell=True,
                    executable="/bin/bash",
                    #         capture_output=True,
                    #         text=True,
                )


def eval_and_store_results(
    files_map: Dict[str, List[str]],
    dataset: str,
    result: Dict[SOTA, Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]],
) -> None:
    for sota in MODELS_CHECKPOINTS:
        if sota != SOTA.FLIP_FAS:
            continue
        if sota not in result:
            result[sota] = {}
        for protocol, ckpt in MODELS_CHECKPOINTS[sota].items():
            if protocol not in result[sota]:
                result[sota][protocol] = {}
            if dataset not in result[sota][protocol]:
                result[sota][protocol][dataset] = {}
            os.makedirs(
                f"./results/{dataset}/{sota.value.lower()}/{protocol}", exist_ok=True
            )
            for exp, files in files_map.items():
                if dataset + "_" + exp not in result[sota][protocol]:
                    result[sota][protocol][dataset][exp] = {"real": [], "attack": []}
                if os.path.isfile(
                    f"./results/{dataset}/{sota.value.lower()}/{protocol}/{exp}.json"
                ):
                    with open(
                        f"./results/{dataset}/{sota.value.lower()}/{protocol}/{exp}.json",
                        "r",
                    ) as fp:
                        scores = json.load(fp)
                    if "result" in scores:
                        scores = scores["result"]

                else:
                    scores = retrieve_result(
                        files, sota.value.lower(), ckpt, exp + ".json"
                    )
                    print("Done:", exp)
                    with open(
                        f"./results/{dataset}/{sota.value.lower()}/{protocol}/{exp}.json",
                        "w+",
                    ) as fp:
                        json.dump({"result": scores}, fp)

                if "real" in exp:
                    result[sota][protocol][dataset][exp]["real"].extend(scores)
                else:
                    result[sota][protocol][dataset][exp]["attack"].extend(scores)


def plot_det_curve(
    genuine: np.ndarray, impostor: np.ndarray, fname: str, bins: int = 10_001
):
    # print(genuine.shape, impostor.shape)
    mi = min(np.min(impostor), np.min(genuine))
    mx = max(np.max(impostor), np.max(genuine))

    # Following 3 lines are optional
    # Normalize the scores
    # impostor = (impostor - mi) / (mx - mi)
    # genuine = (genuine - mi) / (mx - mi)
    thresholds = np.linspace(0, 1, bins)

    far = np.zeros(bins)
    frr = np.zeros(bins)
    for i, threshold in enumerate(thresholds):
        far[i] = np.sum(impostor >= threshold) * 100 / len(impostor)
        frr[i] = np.sum(genuine < threshold) * 100 / len(genuine)

    plt.figure()
    plt.plot(far, frr)
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("FAR")
    plt.ylabel("FRR")
    plt.title("DET Curve")
    plt.savefig(fname)
    plt.close()


if __name__ == "__main__":
    result: Dict[SOTA, Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]] = {}

    # idiap:
    files_map = get_files_for_idiap()
    dataset = "idiap"
    eval_and_store_results(files_map, dataset, result)

    # HDA
    files_map = get_files_for_hda()
    dataset = "hda"
    eval_and_store_results(files_map, dataset, result)

    # nexdata
    files_map = get_files_for_nexdata()
    dataset = "nexdata"
    eval_and_store_results(files_map, dataset, result)

    # ibeta
    files_map = get_files_for_trainingpro2()
    dataset = "trainingproibeta2"
    eval_and_store_results(files_map, dataset, result)

    for sota in MODELS_CHECKPOINTS:
        if sota != SOTA.FLIP_FAS:
            continue
        for protocol, ckpt in MODELS_CHECKPOINTS[sota].items():
            os.makedirs(f"./results/det_{sota.value.lower()}/{protocol}", exist_ok=True)
            for dataset in result[sota][protocol]:
                exp_scores = result[sota][protocol][dataset]
                os.makedirs(
                    f"./results/det_{sota.value.lower()}/{protocol}/{dataset}",
                    exist_ok=True,
                )
                if dataset == "idiap":
                    real_scores = np.array(exp_scores["real_v1"]["real"])
                    for attack in exp_scores:
                        if "real" in attack or "v2" in attack:
                            continue
                        print("idiap v1", attack)
                        attack_scores = np.array(exp_scores[attack]["attack"])
                        plot_det_curve(
                            real_scores,
                            attack_scores,
                            f"./results/det_{sota.value.lower()}/{protocol}/{dataset}/v1_{attack}_det.png",
                        )

                    real_scores = np.array(exp_scores["real_v2"]["real"])
                    for attack in exp_scores:
                        if "real" in attack or "v1" in attack:
                            continue
                        attack_scores = np.array(exp_scores[attack]["attack"])
                        print("idiap v2", attack)
                        plot_det_curve(
                            real_scores,
                            attack_scores,
                            f"./results/det_{sota.value.lower()}/{protocol}/{dataset}/v2_{attack}_det.png",
                        )

                else:
                    real_scores = np.array(exp_scores["real"]["real"])
                    for attack in exp_scores:
                        if "real" in attack:
                            continue
                        print(dataset, attack)
                        attack_scores = np.array(exp_scores[attack]["attack"])
                        plot_det_curve(
                            real_scores,
                            attack_scores,
                            f"./results/det_{sota.value.lower()}/{protocol}/{dataset}/{attack}_det.png",
                        )
