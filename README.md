


### Install Instructions

```
conda activate base

conda remove -n dguafas --all -y
conda remove -n gacdfas --all -y
conda remove -n jpdfas --all -y

cd ./models/DGUA_FAS/
cd ./ml-cvnets
# git clone https://github.com/apple/ml-cvnets
# cd ml-cvnets
# git checkout 84d992f413e52c0468f86d23196efd9dad885e6f
# cp ../base_cls.py ./cvnets/models/classification/base_cls.py
conda create -n dguafas python==3.9.12 -y
conda init
. ~/.bashrc
conda activate dguafas 
     
pip install -r requirements.txt
pip install --editable .
pip install pandas
pip install tensorboard
conda deactivate
cd ..
conda info -e
cd ../..

cd ./models/GACD_FAS/
conda create -n gacdfas python==3.6.12 -y
conda init
. ~/.bashrc
conda activate gacdfas 

pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
pip install easydict

conda deactivate

conda info -e

cd ../..

cd ./models/JPD_FAS/
conda create -n jpdfas python==3.9.12 -y
conda init
. ~/.bashrc
conda activate jpdfas 

pip install einops
pip install -r requirements.txt

conda deactivate

conda info -e

cd ../..

conda clean -a -y

```

### Test all the models: 

```
conda activate dguafas
python test_model.py --model-name=DGUA_FAS --path=./pretrained_models/DGUA_FAS/I&C&MtoO/best_model.pth.tar
conda deactivate
conda activate gacdfas
python test_model.py --model-name=GACD_FAS --path=./pretrained_models/GACD_FAS/resnet18_pICM2O_best.pth
conda deactivate
conda activate jpdfas
python test_model.py --model-name=JPD_FAS --path=./pretrained_models/JPD_FAS/full_resnet50.pth
conda deactivate

```
