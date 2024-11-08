import os

import numpy as np
import pandas as pd

from eval_loop import ATTACKS, MODELS_CHECKPOINTS
from util.metrics import calculate_eer

result = {
    "iPhone": [],
    "protocol": [],
    "attack": [],
    "model": [],
    "train_eer": [],
    "test_eer": [],
}

for model in MODELS_CHECKPOINTS:
    for protocol, ckpt in MODELS_CHECKPOINTS[model].items():
        for iphone in ["iPhone11", "iPhone12"]:
            rdir = f"/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/3D_PAD_Datasets/2D_Face_Databases_PAD/{iphone}/Data_Split/"
            for attack in ATTACKS:
                edir = os.path.join(
                    "./tmp/", model, protocol, "indian_pad", iphone, attack
                )
                if not os.path.isdir(edir):
                    continue

                result["attack"].append(attack)
                result["iPhone"].append(iphone)
                result["protocol"].append(protocol)
                result["model"].append(model)

                try:
                    train_real = np.loadtxt(os.path.join(edir, "train_real.txt"))
                    train_attack = np.loadtxt(os.path.join(edir, "train_attack.txt"))

                    result["train_eer"].append(
                        round(
                            calculate_eer(
                                train_real.squeeze().tolist(),
                                train_attack.squeeze().tolist(),
                            ),
                            4,
                        )
                    )

                except:
                    result["train_eer"].append(None)

                try:
                    test_real = np.loadtxt(os.path.join(edir, "test_real.txt"))
                    test_attack = np.loadtxt(os.path.join(edir, "test_attack.txt"))

                    result["test_eer"].append(
                        round(
                            calculate_eer(
                                test_real.squeeze().tolist(),
                                test_attack.squeeze().tolist(),
                            ),
                            4,
                        )
                    )

                except:
                    result["test_eer"].append(None)

for k, v in result.items():
    print(k, len(v))
df = pd.DataFrame(result)
print(df)
df.to_csv("/mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/pad-bench/eval.csv")
