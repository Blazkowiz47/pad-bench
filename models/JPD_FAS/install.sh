#!/bin/bash

conda init
conda create -n jpdfas python==3.9.12 -y
conda activate jpdfas 

pip install -r requirements.txt

conda deactivate

conda info -e

