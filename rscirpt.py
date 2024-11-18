import os
import shutil
from util import SOTA

print([sota.name for sota in SOTA])
print(SOTA("DGUA_FAS") == SOTA.DGUA_FAS)

exit()

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

rdir = "./tmp/JPD_FAS"
Odir = "./tmp/LMFD_FAS"

for iphone in ["iPhone11", "iPhone12"]:
    for attack in ATTACKS:
        logdir = os.path.join(rdir, iphone, attack)
        ckptdir = os.path.join(rdir, iphone, attack, "checkpoints")
        if os.path.isfile(os.path.join(logdir, "train.txt")) and os.path.isfile(
            os.path.join(ckptdir, "best_weights.pth")
        ):
            odir = os.path.join(Odir, iphone, attack)
            os.makedirs(odir, exist_ok=True)
            os.makedirs(os.path.join(odir, "checkpoints"), exist_ok=True)
            shutil.move(
                os.path.join(logdir, "train.txt"), os.path.join(odir, "train.txt")
            )
            shutil.move(
                os.path.join(ckptdir, "best_weights.pth"),
                os.path.join(odir, "checkpoints", "best_weights.pth"),
            )
