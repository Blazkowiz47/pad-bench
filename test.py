from models import get_model
from util.logger import get_logger

pretrained_models = {
    "DGUA_FAS": "./pretrained_models/DGUA_FAS/I&C&MtoO/best_model.pth.tar",
    "GACD_FAS": "./pretrained_models/GACD_FAS/resnet18_pICM2O_best.pth",
    "JPD_FAS": "./pretrained_models/JPD_FAS/full_resnet50.pth",
}

log = get_logger("./logs/test.log")
model_name = "JPD_FAS"

get_model(
    model_name,
    {"arch": "resnet50"},
    log,
    path=pretrained_models[model_name],
)
