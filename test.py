import os

pretrained_models = {
    "DGUA_FAS": "./pretrained_models/DGUA_FAS/I&C&MtoO/best_model.pth.tar",
    "GACD_FAS": "./pretrained_models/GACD_FAS/resnet18_pICM2O_best.pth",
    "JPD_FAS": "./pretrained_models/JPD_FAS/full_resnet50.pth",
}

for model, path in pretrained_models.items():
    env = model.lower().replace("_", "")
    script: str = f"""
    conda activate {env};
    python test_model.py {model} {path}    
    """
    print(script)
    continue
    os.system(script)
