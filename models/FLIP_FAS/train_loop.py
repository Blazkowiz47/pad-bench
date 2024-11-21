import os

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

if __name__ == "__main__":
    args = []
    for method in ["flip_it"]:
        for iphone in ["iPhone11", "iPhone12"]:
            for attack in reversed(ATTACKS):
                rdir = f"/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/3D_PAD_Datasets/2D_Face_Databases_PAD/{iphone}/Data_Split/"
                edir = f"../../tmp/FLIP_FAS/{method}/{iphone}/{attack}"
                os.makedirs(edir, exist_ok=True)
                ldir = f"../../tmp/FLIP_FAS/{method}/{iphone}/{attack}/train.log"
                if os.path.isfile(ldir):
                    continue
                cmd = f"python train_flip.py \
                      --attack {attack} \
                      --rdir {rdir} \
                      --op_dir '{edir}' \
                      --report_logger_path '{ldir}' \
                      --method {method} \
                    "
                os.system(cmd)
