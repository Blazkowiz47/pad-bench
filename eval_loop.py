import os
import subprocess

BATCH_SIZE = 64
MODELS_CHECKPOINTS = {
    "CF_FAS": {
        "icmo": "./pretrained_models/CF_FAS/icmo.pth",
        "oimc": "./pretrained_models/CF_FAS/omic.pth",
        "ocim": "./pretrained_models/CF_FAS/ocim.pth",
        "ocmi": "./pretrained_models/CF_FAS/ocmi.pth",
    },
    "LMFD_FAS": {
        "icmo": "./pretrained_models/LMFD_FAS/icm_o.pth",
        "oimc": "./pretrained_models/LMFD_FAS/omi_c.pth",
        "ocim": "./pretrained_models/LMFD_FAS/oci_m.pth",
        "ocmi": "./pretrained_models/LMFD_FAS/ocm_i.pth",
    },
    "GACD_FAS": {
        "icmo": "./pretrained_models/GACD_FAS/resnet18_pICM2O_best.pth",
        "oimc": "./pretrained_models/GACD_FAS/resnet18_pOMI2C_best.pth",
        "ocim": "./pretrained_models/GACD_FAS/resnet18_pOCI2M_best.pth",
        "ocmi": "./pretrained_models/GACD_FAS/resnet18_pOCM2I_best.pth",
    },
    "JPD_FAS": {"all": "./pretrained_models/JPD_FAS/full_resnet50.pth"},
    "DGUA_FAS": {
        "icmo": "./pretrained_models/DGUA_FAS/I&C&MtoO/best_model.pth.tar",
        "oimc": "./pretrained_models/DGUA_FAS/O&I&MtoC/best_model.pth.tar",
        "ocim": "./pretrained_models/DGUA_FAS/O&C&ItoM/best_model.pth.tar",
        "ocmi": "./pretrained_models/DGUA_FAS/O&C&MtoI/best_model.pth.tar",
    },
}

ATTACKS = [
    "display",
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


def eval_all_indian_pads() -> None:
    """
    model: model name
    attacK: default = '*', otherwise mention attack name
    ckpt: checkpoint path
    edir: evaluation directory path
    rdir: dataset root directory path
    """
    for model in MODELS_CHECKPOINTS:
        for protocol, ckpt in MODELS_CHECKPOINTS[model].items():
            for iphone in ["iPhone11", "iPhone12"]:
                rdir = f"/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/3D_PAD_Datasets/2D_Face_Databases_PAD/{iphone}/Data_Split/"
                for attack in ATTACKS:
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
                            -edir {edir}  --logger-level=DEBUG'
                    conda_env = model.lower().replace("_", "")
                    subprocess.run(
                        f"{source_call}; conda activate {conda_env}; {syscall}; conda deactivate",
                        shell=True,
                        executable="/bin/bash",
                        #         capture_output=True,
                        #         text=True,
                    )


if __name__ == "__main__":
    eval_all_indian_pads()
