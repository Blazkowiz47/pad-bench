#!/bin/bash

conda init
conda create -n gacdfas python==3.6.12 -y
conda activate gacdfas 

pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install easydict

conda deactivate

conda info -e

