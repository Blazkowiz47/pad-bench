import os

os.system("conda create -n jpdfas python==3.8.10 -y")
os.system("conda init")
os.system(". ~/.bashrc")
os.system("conda activate jpdfas ")

os.system("pip install einops")
os.system("pip install -r requirements.txt")

os.system("conda deactivate")

os.system("conda info -e")
