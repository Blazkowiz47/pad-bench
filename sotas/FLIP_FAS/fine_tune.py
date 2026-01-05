import argparse
import os
import random
from typing import List, Tuple

import numpy as np
from tqdm import tqdm
import torch
from torch.nn import (
    CrossEntropyLoss,
    Module,
)
from torch.optim import AdamW
from transformers import CLIPProcessor, CLIPModel
from torchmetrics.classification import BinaryAccuracy

from morph_desc import morph_list, genuine_list
from dataset import Wrapper

parser = argparse.ArgumentParser()


parser.add_argument(
    "-o",
    "--model-name",
    type=str,
    default="",
)


def forward(imgs, txts, model, processor) -> torch.Tensor:
    inputs = processor(
        text=txts,
        images=imgs,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.cuda() for k, v in inputs.items()}
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    return logits_per_image


def main(args: argparse.Namespace) -> None:
    rdir = "ROOT_DIR"  # noqa: E501
    printer = args.printer
    morph_type = args.morph

    if "," in printer:
        printer = printer.split(",")
    if "," in morph_type:
        morph_type = morph_type.split(",")

    wrapper = Wrapper(rdir, morph_type, printer, 32)
    trainds = wrapper.get_train(batch_size=32)
    testds = wrapper.get_test(batch_size=64)

    model: CLIPModel = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor: CLIPProcessor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-base-patch32"
    )
    for params in model.parameters():
        params.requires_grad = False

    for p in model.text_projection.parameters():
        p.requires_grad = True

    for p in model.visual_projection.parameters():
        p.requires_grad = True

    model.logit_scale.requires_grad = True

    model = model.cuda()
    optimizer = AdamW(
        [param for param in model.parameters() if param.requires_grad],
        1e-5,
    )
    loss_fn = CrossEntropyLoss().cuda()
    best_train_loss: float = float("inf")
    best_test_accuracy: float = 0.0
    metric = BinaryAccuracy()

    step_breaker = 1000
    step = 0

    for epoch in range(100):
        model.train()
        print("Epoch:", epoch)
        total_loss = 0.0
        for imgs, lbls in tqdm(trainds, desc="Training"):
            model.train()
            step += 1
            optimizer.zero_grad()
            imgs, lbls = imgs, lbls.cuda()

            text = [random.choice(morph_list), random.choice(genuine_list)]
            inputs = processor(
                text=text,
                images=imgs,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.cuda() for k, v in inputs.items()}
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(
                dim=1
            )  # we can take the softmax to get the label probabilities

            metric.update(probs.argmax(dim=1), lbls.argmax(dim=1))
            loss = loss_fn(logits_per_image, lbls)
            loss.backward()
            optimizer.step()

            total_loss += loss.detach().cpu().item()

            if not step % step_breaker:
                print("Train Loss:", total_loss)
                accuracy = metric.compute()
                if best_train_loss > total_loss:
                    os.makedirs("checkpoints", exist_ok=True)
                    best_train_loss = total_loss
                    torch.save(
                        model.state_dict(),
                        f"./checkpoints/step_{step}_{accuracy.detach().cpu().item():0.3}.pt",
                    )

                model.eval()
                total_loss = 0.0
                for imgs, lbls in tqdm(testds):
                    imgs, lbls = imgs, lbls.cuda()
                    [random.choice(morph_list), random.choice(genuine_list)]

        accuracy = (bon_correct + mor_correct) / (bon_incorrect + mor_incorrect)
        if accuracy > best_test_accuracy:
            best_test_accuracy = accuracy
            torch.save(
                model.state_dict(),
                f"./checkpoints/{args.model_name or ('best_cross_domain_' + morph_type + '_accuracy_model')}.pt",  # noqa: E501
            )


def set_seeds(seed: int = 2024):
    """Sets random sets for torch operations.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


if __name__ == "__main__":
    args = parser.parse_args()

    main(args)
