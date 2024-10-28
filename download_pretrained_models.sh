#!/bin/bash

conda init
conda activate base 
pip install gdown 

cd ./pretrained_models

# gdown --folder https://drive.google.com/drive/folders/1D8WZjO62Kv4uzzNouzJWs2BrBayZq_0l -O DGUA_FAS

mkdir JPDFAS 
cd ./JPDFAS 
wget https://download.pytorch.org/models/resnet50-19c8e357.pth
gdown https://drive.google.com/uc?id=1gNUMzaK87CcQlM_aCr7TjU3iLx_2NiBp
gdown https://drive.google.com/uc?id=1pTxdI2uZj1yBSCN64vKs-cQddIZN6rvO
gdown https://drive.google.com/uc?id=1GTAoTmyCbYGws5BV07qUL-iGVvHx_LmF
gdown https://drive.google.com/uc?id=1VpWN8CXdVVLTwyTPABeFXmr3UnnenjYe
gdown https://drive.google.com/uc?id=1Ii3JmoRjWcOLF4xNwCqtJyp0Ok0vJva3
gdown https://drive.google.com/uc?id=1E4UD8UK_KzjhpAvR6hYInlteOEaxDZbZ
gdown https://drive.google.com/uc?id=1UhCrC2VQCz4zE1UFc-lnziivqB3bSIpP
gdown https://drive.google.com/uc?id=18e23EW2ncsnOqET4jCSEskHVNvFYAF1j
gdown https://drive.google.com/uc?id=16VzZSYcVTNErFoLF87urdrzQbGbKvqjX

