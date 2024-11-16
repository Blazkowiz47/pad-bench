import os
import re
from glob import glob
from eval_loop import ATTACKS
from common_metrics import eer 


def get_checkpoint_path(sota: str, iphone: str, attack: str) -> str:
    if sota == "JPD_FAS":
        models = glob(f"./tmp/JPD_FAS/{iphone}/{attack}/checkpoints/resnet50_*.pth")
        best_accuracy = 0
        best_model = None
        for model in models:
            _, tail = os.path.split(model)
            accuracy = float(tail.split("_")[-1].replace(".pth", ""))
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
        if not best_model:
            raise ValueError(f"JPD_FAS not trained for {iphone} and {attack}")
        return best_model

    if sota == "LMFD_FAS":
        return f"./tmp/LMFD_FAS/{iphone}/{attack}/checkpoints/best_weights.pth"

    if sota == "CF_FAS":
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

                print("Epoch number:", epoch_number)
                print("loss_total value:", loss_total_value)
        if not best_model:
            raise ValueError(f"CF_FAS not trained for {iphone} and {attack}")

        return best_model

    raise ValueError(f"SOTA: {sota} not trained for {iphone} and {attack}")

def eval_loop() -> None: 
    for sota in ['JPD_FAS', 'CF_FAS', 'LMFD_FAS']:
        for iphone in ['iPhone11', 'iPhone12']:
            for attack in ATTACKS:



