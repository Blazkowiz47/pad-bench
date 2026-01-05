#!/bin/bash

conda create -n jpdfas python==3.9.12 -y
conda init
. ~/.bashrc
conda activate jpdfas 

pip install einops
pip install -r requirements.txt

conda deactivate

conda info -e

