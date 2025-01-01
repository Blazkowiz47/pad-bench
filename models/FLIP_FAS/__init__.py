from typing import Any, Dict, List, Optional
from logging import Logger

from PIL import Image
import torch
from torch.nn import Module
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .fas import flip_mcl


def get_model(config: Dict[str, Any], log: Logger, **kwargs) -> Module:
    log.debug("Initialising FLIP_FAS Model")
    log.debug(f"Provided config: {config}")
    net = flip_mcl(
        in_dim=512, ssl_mlp_dim=4096, ssl_emb_dim=256
    )  # ssl applied to image, and euclidean distance applied to image and text cosine similarity

    log.debug("Initialised FLIP_FAS Model")
    if kwargs.get("path", ""):
        net_ = torch.load(kwargs["path"])
        net.load_state_dict(net_["state_dict"])
        log.debug("Loaded weights of FLIP_FAS Model")

    return net


def get_scores(
    data_loader: DataLoader, model: Module, log: Logger, position: Optional[int] = 0
) -> Dict[str, List[float]]:
    result: Dict[str, List[float]] = {"attack": [], "real": []}

    model.eval()
    model.cuda()
    for x, y in tqdm(data_loader, position=position):
        x = x.cuda()
        y = y.argmax(dim=1).numpy().tolist()
        cls_out, _ = model.forward_eval(x)
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
