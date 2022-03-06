FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu18.04


ARG PYTHON_VERSION=3.8
ARG WITH_TORCHVISION=1

# Install necessary building tools and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         git \
         curl \
         sudo \
         bzip2 \
         libx11-6 \
         ca-certificates \
         libjpeg-dev \
         libpng-dev && \
     rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get -y dist-upgrade && apt-get purge -y libboost-all-dev && \
    apt-get install -f -y libboost-all-dev && \
    rm -rf /var/lib/apt/lists/*


# Create a working directory
RUN mkdir /app
WORKDIR /app


# Install conda and torch
ENV PATH=/opt/conda/bin:$PATH
RUN curl -o ~/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=$PYTHON_VERSION numpy pyyaml scipy ipython mkl mkl-include ninja cython typing && \
    /opt/conda/bin/conda clean -ya

# https://github.com/anibali/docker-pytorch/blob/master/dockerfiles/1.7.0-cuda11.0-ubuntu20.04/Dockerfile
ARG CUDATOOLKIT_VERSION=11.0
ENV CONDA_OVERRIDE_CUDA=11.0
RUN /opt/conda/bin/conda install -y -c pytorch \
    cudatoolkit=11.0.221 \
    "pytorch=1.7.0=py3.8_cuda11.0.221_cudnn8.0.3_0" \
    "torchvision=0.8.1=py38_cu110" \
 && /opt/conda/bin/conda clean -ya

ENV PATH /opt/conda/bin:$PATH

RUN apt-get update
RUN apt-get install -y vim lshw
RUN apt-get install wget libssl-dev

ENV LANG C.UTF-8
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install data science and machine learning packages using conda
RUN /opt/conda/bin/conda install -y scikit-learn pandas seaborn && \
    /opt/conda/bin/conda clean -ya

RUN /opt/conda/bin/conda install -y -c conda-forge matplotlib tensorboard && \
    /opt/conda/bin/conda install -c anaconda python-dateutil && \
    /opt/conda/bin/conda clean -ya
