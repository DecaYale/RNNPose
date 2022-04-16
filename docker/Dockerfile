FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

# Dependencies for glvnd and X11.
RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libxext6 \
    libx11-6 \
  && rm -rf /var/lib/apt/lists/*
# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

#env vars for cuda
ENV CUDA_HOME /usr/local/cuda

#install miniconda
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN wget --quiet https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py37_4.9.2-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/miniconda3 && \
    rm ~/miniconda.sh && \
    /opt/miniconda3/bin/conda clean -tipsy && \
    ln -s /opt/miniconda3/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    echo "conda deactivate && conda activate py37" >> ~/.bashrc

#https://blog.csdn.net/Mao_Jonah/article/details/89502380
COPY freeze.yml freeze.yml
RUN /opt/miniconda3/bin/conda env create -n py37 -f freeze.yml

# WORKDIR /tmp/
# COPY config.jupyter.tar config.jupyter.tar
# RUN tar -xvf config.jupyter.tar -C /root/

#install apex
ENV TORCH_CUDA_ARCH_LIST "6.0 6.2 7.0 7.2"
# make sure we don't overwrite some existing directory called "apex"
WORKDIR /tmp/unique_for_apex
# uninstall Apex if present, twice to make absolutely sure :)
RUN /opt/miniconda3/envs/py37/bin/pip3 uninstall -y apex || :
RUN /opt/miniconda3/envs/py37/bin/pip3 uninstall -y apex || :
# SHA is something the user can touch to force recreation of this Docker layer,
# and therefore force cloning of the latest version of Apex
RUN SHA=ToUcHMe git clone https://github.com/NVIDIA/apex.git
WORKDIR /tmp/unique_for_apex/apex
RUN /opt/miniconda3/envs/py37/bin/pip3 install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
#install pytorch3d 
# RUN /opt/miniconda3/envs/py37/bin/pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py37_cu102_pyt171/download.html
# RUN /opt/miniconda3/envs/py37/bin/pip install "git+https://github.com/facebookresearch/pytorch3d.git"
# RUN /opt/miniconda3/bin/conda install pytorch3d==0.5.0 -c pytorch3d -n py37



#other pkgs
RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
  cmake build-essential vim xvfb unzip tmux psmisc  \
  libx11-dev libassimp-dev \
  mesa-common-dev freeglut3-dev \
  rsync \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

#create some directories
RUN mkdir -p /home/RNNPose

EXPOSE 8887 8888 8889 10000 10001 10002 
WORKDIR /home/RNNPose

