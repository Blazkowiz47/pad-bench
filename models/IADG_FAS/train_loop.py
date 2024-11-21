from multiprocessing import Pool
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
    port = 12345
    for iphone in ["iPhone12", "iPhone11"]:
        for attack in ATTACKS:
            rdir = f"/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/3D_PAD_Datasets/2D_Face_Databases_PAD/{iphone}/Data_Split/"
            edir = f"../../tmp/IADG_FAS//{iphone}/{attack}"
            cmd = f"python -m torch.distributed.run --nproc_per_node=1 --master_port={port} \
                  train.py \
                  --attack {attack} \
                  --rdir {rdir} \
                  --exam_dir '{edir}' \
                "
            port += 1
            args.append(cmd)

    with Pool(3) as p:
        p.map(os.system, args)
