import subprocess
from util import SOTA


for sota in SOTA:
    conda_env = sota.name
    script: str = f"""
    source ~/miniconda3/etc/profile.d/conda.sh;
    conda activate {conda_env.lower().replace('_','')};
    python -c 'from logging import getLogger
import numpy
import torch
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from models import get_model, get_score_function, get_transform_function
from util import SOTA
import yaml
log = getLogger()
with open("./configs/train.yaml", "r") as fp:
    config = yaml.safe_load(fp)
image_x = torch.randn(1, 3, 224, 224).cuda()
model = get_model(SOTA.{conda_env}, config, log).cuda().eval()
flops = FlopCountAnalysis(model, image_x)
total_flops = flops.total()
print("FLOPs (G) for model: {conda_env}:", total_flops / 1_000_000_000)';
    conda deactivate;
    """
    output = subprocess.run(
        script, shell=True, executable="/bin/bash", capture_output=True
    )
    print("output: ", output.stdout)
