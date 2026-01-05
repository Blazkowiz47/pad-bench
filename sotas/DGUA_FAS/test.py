import torch
from logging import getLogger
from model import get_model


def driver() -> None:
    log = getLogger()
    model = get_model(
        {"config_path": "./configs/mobilevit_s.yaml"},
        log,
        path="/root/pretrained_models/DGUA_FAS/I&C&MtoO/best_model.pth.tar",
    )
    model.cuda().eval()
    y = model(torch.randn(1, 3, 224, 224).cuda())
    print(y.shape)


if __name__ == "__main__":
    driver()
