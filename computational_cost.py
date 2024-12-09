import subprocess
from logging import getLogger

import torch
import yaml
from fvcore.nn import FlopCountAnalysis
from torch.nn import Module

from models import get_model
from util import SOTA

for sota in SOTA:
    if sota == SOTA.IADG_FAS:
        continue

    conda_env = sota.name
    script: str = f"""
    source ~/miniconda3/etc/profile.d/conda.sh;
    conda activate {conda_env.lower().replace('_','')};
    python -c 'from logging import getLogger
import numpy
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from models import get_model
from util import SOTA
import yaml
log = getLogger()
with open("./configs/train.yaml", "r") as fp:
    config = yaml.safe_load(fp)
image_x = torch.randn(1, 3, 224, 224).cuda()
model = get_model(SOTA.{conda_env}, config, log).cuda().eval()
flops = FlopCountAnalysis(model, image_x)
total_flops = flops.total()
print("FLOPs (G) for model {conda_env}:", total_flops / 1_000_000_000)
print("Total params (1e6):", sum(p.numel() for p in model.parameters()) / 1_000_000)
';
    conda deactivate;
    """
    output = subprocess.run(
        script, shell=True, executable="/bin/bash", capture_output=True
    )
    print(output.stdout.decode())


log = getLogger()
with open("./configs/train.yaml", "r") as fp:
    config = yaml.safe_load(fp)
image_x = torch.randn(1, 3, 256, 256).cuda()


class DemoMod(Module):
    def __init__(self) -> None:
        super(DemoMod, self).__init__()
        self.net = get_model(SOTA.IADG_FAS, config, log).cuda().eval()

    def forward(self, x):
        outputs_catcls, outputs_catdepth, *_ = self.net(x, None, True, False, False)
        bs, _, _, _ = outputs_catdepth["out"].shape
        probs = torch.softmax(outputs_catcls["out"], dim=1)[:, 0]
        depth_probs = outputs_catdepth["out"].reshape(bs, -1).mean(dim=1)
        pred = probs + depth_probs

        return pred


model = DemoMod().cuda().eval()
flops = FlopCountAnalysis(model, image_x)
total_flops = flops.total()
print("FLOPs (G) for model IADG_FAS:", total_flops / 1_000_000_000)
print("Total params (1e6):", sum(p.numel() for p in model.parameters()) / 1_000_000)
