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
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio
RUN pip install fvcore
RUN pip install einops
RUN pip install timm

# SOTA specific dependencies
RUN pip install thop

# Keep container alive:
CMD ["/bin/bash"]

