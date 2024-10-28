#!/bin/bash

cd ./ml-cvnets
# git clone https://github.com/apple/ml-cvnets
# cd ml-cvnets
# git checkout 84d992f413e52c0468f86d23196efd9dad885e6f
# cp ../base_cls.py ./cvnets/models/classification/base_cls.py

conda init
conda create -n dguafas python==3.9.12 -y
conda activate dguafas 
     
pip install -r requirements.txt
pip install --editable .
pip install pandas
pip install tensorboard
conda deactivate
cd ..

conda info -e

