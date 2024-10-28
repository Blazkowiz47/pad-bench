import os

os.system("conda remove -n dguafas --all -y")
os.system("conda remove -n gacdfas --all -y")
os.system("conda remove -n jpdfas --all -y")


os.chdir("./models/DGUA_FAS/")
os.system("python install.py")
os.chdir("../..")

os.chdir("./models/GACD_FAS/")
os.system("python install.py")
os.chdir("../..")

os.chdir("./models/JPD_FAS/")
os.system("python install.py")
os.chdir("../..")


os.system("conda clean -a -y")
