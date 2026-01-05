FROM python:3.8.10
WORKDIR /root/code

ENV TERM=xterm-256color

# Dependencies
RUN pip install --upgrade pip
RUN pip install numpy==1.24.4
RUN pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
RUN pip install fvcore
RUN pip install einops
RUN pip install timm
RUN pip install albumentations
RUN pip install omegaconf


CMD ["/bin/bash"]
