### Install Instructions

```
ln -s /mnt/cluster/nbl-users/Shreyas-Sushrut-Raghu/pad-bench/pretrained_models /home/ubuntu/pad-bench/pretrained_models
conda activate base

conda remove -n dguafas --all -y
conda remove -n gacdfas --all -y
conda remove -n jpdfas --all -y
conda remove -n cffas --all -y
conda remove -n lmfdfas --all -y
conda remove -n flipfas --all -y

cd ./models/DGUA_FAS/
conda env create --file=environment.yml -y
conda info -e
cd ../..

cd ./models/GACD_FAS/
conda env create --file=environment.yml -y
conda info -e
cd ../..

cd ./models/JPD_FAS/
conda env create --file=environment.yml -y
conda info -e
cd ../..

cd ./models/CF_FAS/
conda env create --file=environment.yml -y
conda info -e
cd ../..

cd ./models/LMFD_FAS/
conda env create --file=environment.yml -y
conda info -e
cd ../..

cd ./models/FLIP_FAS/
conda env create --file=environment.yml -y
conda info -e
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
conda activate lmfdfas
python test_model.py --model-name=LMFD_FAS --path="./pretrained_models/LMFD_FAS/icm_o.pth"
conda deactivate
conda activate flipfas
python test_model.py --model-name=FLIP_FAS --path="./pretrained_models/FLIP_FAS/msu_flip_mcl.pth.tar"
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
