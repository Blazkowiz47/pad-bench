FROM python:3.8.20
WORKDIR /root/code

# Personal Setup for developemnt inside the container
RUN curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim-linux-x86_64.appimage
RUN chmod u+x nvim-linux-x86_64.appimage
RUN ./nvim-linux-x86_64.appimage --appimage-extract
RUN ./squashfs-root/AppRun --version

RUN mv squashfs-root /
RUN ln -s /squashfs-root/AppRun /usr/bin/nvim

RUN apt update
RUN apt install -y tmux
RUN git clone https://github.com/Blazkowiz47/nvim-config.git /root/.config/nvim
RUN git clone https://github.com/github/copilot.vim.git ~/.config/nvim/pack/github/start/copilot.vim
RUN apt install luarocks npm ripgrep fd-find -y
RUN luarocks install jsregexp
RUN npm cache clean -f
RUN npm install -g n
RUN n stable
RUN npm install -g tree-sitter-cli
RUN ln -s /root/.config/nvim/.tmux.conf /root/.tmux.conf
RUN echo 'alias ta="tmux attach"' >> /root/.bashrc
RUN apt install python3-venv -y
ENV TERM=xterm-256color

# Dependencies
RUN pip install --upgrade pip --no-cache-dir
RUN pip install torch torchvision --no-cache-dir
RUN pip install fvcore --no-cache-dir
RUN pip install einops --no-cache-dir
RUN pip install timm==0.6.12 --no-cache-dir

# SOTA specific dependencies
RUN pip install thop --no-cache-dir
RUN pip install joblib --no-cache-dir
RUN pip install matplotlib --no-cache-dir
RUN pip install onnx --no-cache-dir
RUN pip install numpy --no-cache-dir
RUN pip install opencv_python --no-cache-dir
RUN pip install pandas --no-cache-dir
RUN pip install Pillow --no-cache-dir
RUN pip install PyYAML --no-cache-dir
RUN pip install requests --no-cache-dir
RUN pip install scipy --no-cache-dir
RUN pip install seaborn --no-cache-dir
RUN pip install tqdm --no-cache-dir
RUN pip install insightface --no-cache-dir
RUN pip install elasticdeform==0.5.0 --no-cache-dir
RUN pip install scikit-image --no-cache-dir
RUN pip install albumentations --no-cache-dir
RUN pip install facenet_pytorch --no-cache-dir

# Keep container alive:
CMD ["/bin/bash"]

