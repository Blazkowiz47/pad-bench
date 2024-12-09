import os
import re
from glob import glob
import subprocess
from multiprocessing import Pool
from typing import List
from util import SOTA

BATCH_SIZE = 64
MODELS_CHECKPOINTS = {
    SOTA.CF_FAS: {
        "icmo": "./pretrained_models/CF_FAS/icmo.pth",
        "oimc": "./pretrained_models/CF_FAS/omic.pth",
        "ocim": "./pretrained_models/CF_FAS/ocim.pth",
        "ocmi": "./pretrained_models/CF_FAS/ocmi.pth",
    },
    SOTA.LMFD_FAS: {
        "icmo": "./pretrained_models/LMFD_FAS/icm_o.pth",
        "oimc": "./pretrained_models/LMFD_FAS/omi_c.pth",
        "ocim": "./pretrained_models/LMFD_FAS/oci_m.pth",
        "ocmi": "./pretrained_models/LMFD_FAS/ocm_i.pth",
    },
    SOTA.GACD_FAS: {
        "icmo": "./pretrained_models/GACD_FAS/resnet18_pICM2O_best.pth",
        "oimc": "./pretrained_models/GACD_FAS/resnet18_pOMI2C_best.pth",
        "ocim": "./pretrained_models/GACD_FAS/resnet18_pOCI2M_best.pth",
        "ocmi": "./pretrained_models/GACD_FAS/resnet18_pOCM2I_best.pth",
    },
    SOTA.JPD_FAS: {"all": "./pretrained_models/JPD_FAS/full_resnet50.pth"},
    SOTA.DGUA_FAS: {
        "icmo": "./pretrained_models/DGUA_FAS/I&C&MtoO/best_model.pth.tar",
        "oimc": "./pretrained_models/DGUA_FAS/O&I&MtoC/best_model.pth.tar",
        "ocim": "./pretrained_models/DGUA_FAS/O&C&ItoM/best_model.pth.tar",
        "ocmi": "./pretrained_models/DGUA_FAS/O&C&MtoI/best_model.pth.tar",
    },
    SOTA.FLIP_FAS: {
        "icmo": "./pretrained_models/FLIP_FAS/oulu_flip_mcl.pth.tar",
        "oimc": "./pretrained_models/FLIP_FAS/casia_flip_mcl.pth.tar",
        "ocim": "./pretrained_models/FLIP_FAS/msu_flip_mcl.pth.tar",
        "ocmi": "./pretrained_models/FLIP_FAS/replay_flip_mcl.pth.tar",
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


def tbiom_2d_exp() -> None:
    """
    model: model name
    attacK: default = '*', otherwise mention attack name
    ckpt: checkpoint path
    edir: evaluation directory path
    rdir: dataset root directory path
    """
    args: List[str] = []
    for sota in [SOTA.JPD_FAS, SOTA.CF_FAS, SOTA.LMFD_FAS, SOTA.IADG_FAS]:
        #     for sota in ["LMFD_FAS"]:
        if sota != SOTA.IADG_FAS:
            continue
        for trained_on_iphone in ["iPhone12", "iPhone11"]:
            for trained_on_attack in ATTACKS:
                for iphone in ["iPhone11", "iPhone12"]:
                    rdir = f"/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/3D_PAD_Datasets/2D_Face_Databases_PAD/{iphone}/Data_Split"
                    for attack in ATTACKS:
                        ckpt = get_checkpoint_path(
                            sota, trained_on_iphone, trained_on_attack
                        )
                        edir = os.path.join(
                            f"/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/3D_PAD_Datasets/2dresults/{sota.name}/trained_on_{trained_on_iphone}_{trained_on_attack}/test_on_{iphone}_{attack}"
                        )
                        #                         if os.path.isfile(
                        #                             os.path.join(edir, "test_attack.txt")
                        #                         ) and os.path.isfile(os.path.join(edir, "test_real.txt")):
                        #                             continue
                        #
                        #                         if not os.path.isdir(os.path.join(rdir, "attack", attack)):
                        #                             continue
                        #                         if check_dir(edir):
                        #                             print("Skipping:", edir)
                        #                             continue

                        source_call = "source ~/miniconda3/etc/profile.d/conda.sh"

                        syscall = f'python evaluation.py -m {sota.name} \
                                -c "configs/train.yaml" -d standard \
                                --attack={attack} --rdir={rdir} \
                                --batch-size={BATCH_SIZE} -ckpt "{ckpt}" \
                                -edir {edir}  --logger-level=DEBUG'
                        conda_env = sota.name.lower().replace("_", "")
                        call = f"{source_call}; conda activate {conda_env}; {syscall}; conda deactivate"
                        args.append(call)

    with Pool(1) as p:
        p.map(call_proc, args)


def eval_all_indian_pads() -> None:
    """
    model: model name
    attacK: default = '*', otherwise mention attack name
    ckpt: checkpoint path
    edir: evaluation directory path
    rdir: dataset root directory path
    """
    for iphone in ["iPhone11", "iPhone12"]:
        for attack in ATTACKS:
            for sota in MODELS_CHECKPOINTS:
                model = sota.name
                for protocol, ckpt in MODELS_CHECKPOINTS[sota].items():
                    rdir = f"/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/3D_PAD_Datasets/2D_Face_Databases_PAD/{iphone}/Data_Split/"
                    edir = os.path.join(
                        "./tmp/", model, protocol, "indian_pad", iphone, attack
                    )

                    if not os.path.isdir(os.path.join(rdir, "attack", attack)):
                        continue
                    if check_dir(edir):
                        print("Skipping:", edir)
                        continue

                    source_call = "source ~/miniconda3/etc/profile.d/conda.sh"

                    syscall = f'python evaluation.py -m {model} \
                            -c "configs/train.yaml" -d standard \
                            --attack={attack} --rdir={rdir} \
                            --batch-size={BATCH_SIZE} -ckpt "{ckpt}" \
                            -edir {edir}  --logger-level=ERROR'
                    conda_env = model.lower().replace("_", "")
                    subprocess.run(
                        f"{source_call}; conda activate {conda_env}; {syscall}; conda deactivate",
                        shell=True,
                        executable="/bin/bash",
                        #         capture_output=True,
                        #         text=True,
                    )


if __name__ == "__main__":
    #     eval_all_indian_pads()
    tbiom_2d_exp()
