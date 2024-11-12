### Install Instructions

```
ln -s /mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/pad-bench/pretrained_models /home/ubuntu/pad-bench/pretrained_models
conda activate base

conda remove -n dguafas --all -y
conda remove -n gacdfas --all -y
conda remove -n jpdfas --all -y

cd ./models/DGUA_FAS/
cd ./ml-cvnets

conda create -n dguafas python==3.9.12 -y
conda activate dguafas 
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch -y
pip install -r requirements.txt
pip install --editable .
pip install pandas
pip install tensorboard
pip install onnxscript
pip install --force-reinstall -v "numpy==1.25.2"
conda deactivate
cd ..
conda info -e
cd ../..

cd ./models/GACD_FAS/
conda create -n gacdfas python==3.6.12 -y
conda activate gacdfas 
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y
pip install easydict
pip install tqdm 
pip install pyyaml
pip install opencv-python
conda deactivate
conda info -e
cd ../..

cd ./models/JPD_FAS/
conda create -n jpdfas -y
conda activate jpdfas 
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt
conda install numpy==1.26.4 -y
conda install termcolor -y
conda deactivate
conda info -e
cd ../..

cd ./models/CF_FAS/
conda create -n cffas -y 
conda activate cffas
conda install pytorch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 pytorch-cuda=12.4 -c pytorch -c nvidia -y
pip install albumentations
conda install tqdm -y 
conda install -c conda-forge scikit-learn -y 
conda deactivate 
conda info -e
cd ../..

cd ./models/LMFD_FAS/
conda create -n lmfdfas --clone cffas -y 
conda activate lmfdfas
cd ../..


conda clean -a -y
```

### Test all the models: 

```
conda activate dguafas
python test_model.py --model-name=DGUA_FAS --path="./pretrained_models/DGUA_FAS/I&C&MtoO/best_model.pth.tar"
conda deactivate
conda activate gacdfas
python test_model.py --model-name=GACD_FAS --path="./pretrained_models/GACD_FAS/resnet18_pICM2O_best.pth"
conda deactivate
conda activate jpdfas
python test_model.py --model-name=JPD_FAS --path="./pretrained_models/JPD_FAS/full_resnet50.pth"
conda deactivate
```

### Get real and attack scores: 

General format: 
```
python -m DGUA_FAS --rdir="/path/to/dataset/" -ckpt "/path/to/pretrained/models" -edir "/path/to/director/to/store/scores/"
```

Example: 
```
python evaluation.py -m DGUA_FAS --rdir="/cluster/nbl-users/Shreyas-Sushrut-Raghu/PAD_Survillance_DB/J7_NG/" \
-ckpt "./pretrained_models/DGUA_FAS/I&C&MtoO/best_model.pth.tar" -edir "./tmp/DGUA_FAS/pad_surveillance/j7ng"
```
