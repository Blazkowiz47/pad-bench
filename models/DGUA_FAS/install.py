import os


os.chdir("./ml-cvnets")
# git clone https://github.com/apple/ml-cvnets
# cd ml-cvnets
# git checkout 84d992f413e52c0468f86d23196efd9dad885e6f
# cp ../base_cls.py ./cvnets/models/classification/base_cls.py

os.system("conda create -n dguafas python==3.9.12 -y")
os.system("conda init")
os.system(". ~/.bashrc")
os.system("conda activate dguafas ")

os.system("pip install -r requirements.txt")
os.system("pip install --editable .")
os.system("pip install pandas")
os.system("pip install tensorboard")
os.system("conda deactivate")
os.chdir("..")

os.system("conda info -e")
