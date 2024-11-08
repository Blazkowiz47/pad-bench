from logging import Logger
from typing import Any, Dict, List, Tuple

import cvnets
from PIL import Image
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torchvision import transforms as T
from tqdm import tqdm


from .option import get_training_arguments


def finetune_epoch(
    model: Module,
    dataset: DataLoader,
    optimizer: Optimizer,
    loss_fn: Module,
    log: Logger,
    device: str,
) -> Tuple[float, float]:
    """
    Returns epoch-loss and accuracy
    """

    for images, labels in tqdm(dataset, leave=True, position=1):
        images, labels = images.to(device), labels.to(device)
        output = model(images)
        log.debug(f"Output of model: {output.shape}")
        log.debug(f"Label of model: {output.shape}")
        exit()

        loss_fn()


def get_model(config: Dict[str, Any], log: Logger, **kwargs) -> Module:
    log.debug("Initalising DGUA_FAS Model")
    log.debug(f"Provided config: {config}")

    opts = get_training_arguments(
        config_path="./models/DGUA_FAS/configs/mobilevit_s.yaml"
    )
    net = cvnets.get_model(opts)

    if not isinstance(net, Module):
        raise ValueError("Cannot load DGUA_FAS model")

    log.debug("Initialised DGUA_FAS Model")
    if kwargs.get("path", ""):
        net_ = torch.load(kwargs["path"])
        net.load_state_dict(net_["state_dict"])
        log.debug("Loaded weights of DGUA_FAS Model")

    return net


def get_scores(
    data_loader: DataLoader, model: Module, log: Logger, position=0
) -> Dict[str, List[float]]:
    result: Dict[str, List[float]] = {"attack": [], "real": []}

    model.eval()
    model.cuda()
    for x, y in tqdm(data_loader, position=position):
        x = x.cuda()
        y = y.argmax(dim=1).numpy().tolist()
        cls_out = model(x, True)[0]
        pred = F.softmax(cls_out, dim=1).detach().cpu().numpy()[:, 1]
        for prob, lbl in zip(pred, y):
            if lbl:
                result["attack"].append(prob.item())
            else:
                result["real"].append(prob.item())

    return result


def transform_image(fname: str) -> torch.Tensor:
    transform = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    img = Image.open(fname).resize((256, 256))
    return transform(img)
