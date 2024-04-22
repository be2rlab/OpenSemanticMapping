FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel AS base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt install -y git-all

WORKDIR /opt/src

COPY ${ROOT_DIR}/openmask3d ./



RUN pip install "Cython<3.0" "pyyaml<6" --no-build-isolation

ARG TORCH_ARCH="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
ENV TORCH_CUDA_ARCH_LIST "${TORCH_ARCH}"
ENV CUDA_HOME /usr/local/cuda-11.6/

RUN bash ./install_requirements.sh
RUN pip install -e .
RUN pip install torchtext==0.13.1
RUN apt install git-lfs


CMD ["bash"]

