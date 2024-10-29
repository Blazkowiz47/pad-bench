import argparse
import torch
import torch.nn.functional as F
from models import get_model, get_score_function, get_transform_function
from util.logger import get_logger


def driver(model_name: str, path: str) -> None:
    log = get_logger("./logs/test.log")
    transform_image = get_transform_function(model_name)
    model = get_model(
        model_name,
        {"arch": "resnet50"},
        log,
        path=path,
    )
    model.cuda().eval()

    img_path = "/cluster/nbl-users/Shreyas-Sushrut-Raghu/FaceMoprhingDatabases/cleaned_datasets/feret/digital/bonafide/test/00201_0.jpg"
    x = transform_image(img_path)
    x = x.unsqueeze(0).cuda()
    print(x.shape)
    _, y = model(x)
    #     y = model(x)
    pred = F.softmax(y, dim=1)[:, 0]
    for prob in pred.detach().cpu().numpy():
        print(prob.item())

    print(y)
    if isinstance(y, tuple):
        for y1 in y:
            print(f"Output of {model_name}:", y1.shape)

    else:
        print(f"Output of {model_name}:", y.shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="JPD_FAS")
    parser.add_argument(
        "--path",
        type=str,
        default="./pretrained_models/JPD_FAS/full_resnet50.pth",
    )
    args = parser.parse_args()
    driver(args.model_name, args.path)
