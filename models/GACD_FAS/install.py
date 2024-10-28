import os

os.system("conda create -n gacdfas python==3.6.12 -y")
os.system("conda init")
os.system(". ~/.bashrc")
os.system("conda activate gacdfas ")

os.system(
    "conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y"
)
os.system("pip install easydict")


os.system("conda deactivate")

os.system("conda info -e")
