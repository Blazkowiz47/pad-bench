import argparse
from models import get_model
from util.logger import get_logger


def driver(model_name: str, path: str) -> None:
    log = get_logger("./logs/test.log")
    get_model(
        model_name,
        {"arch": "resnet50"},
        log,
        path=path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--path", type=str)
    args = parser.parse_args()
    driver(args.model_name, args.path)
