FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US:en \
    LC_ALL=en_US.UTF-8 \
    USER_NAME=aicrowd \
    HOME_DIR=/home/aicrowd \
    CONDA_DIR=/home/aicrowd/.conda \
    PATH=/home/aicrowd/.conda/bin:${PATH} \
    SHELL=/bin/bash

# Install system dependencies and clean up in one layer
COPY apt.txt /tmp/apt.txt
RUN apt -qq update && apt -qq install -y --no-install-recommends `cat /tmp/apt.txt | tr -d '\r'` locales wget build-essential \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/cache/apt/* /var/lib/apt/lists/* \
    && apt clean

# USER ${USER_NAME}
WORKDIR ${HOME_DIR}

# Install Miniconda and Python packages. You can change the python version by using another Miniconda. 
RUN wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" \
    && bash Miniforge3.sh -b -p "${CONDA_DIR}" \
    && . "${CONDA_DIR}/etc/profile.d/conda.sh" \
    && . "${CONDA_DIR}/etc/profile.d/mamba.sh" \
    && conda install cmake -y \
    && conda clean -y -a \
    && rm -rf Miniforge3.sh

COPY --chown=1001:1001 requirements.txt ${HOME_DIR}/requirements.txt
RUN pip install -r requirements.txt --no-cache-dir

COPY --chown=1001:1001 . .
