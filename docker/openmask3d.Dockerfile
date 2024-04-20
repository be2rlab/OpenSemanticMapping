FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel AS base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt install -y git-all

WORKDIR /opt/src

COPY ${ROOT_DIR}/openmask3d ./

ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
ENV PYTHONPATH "${PYTHONPATH}:/opt/src"
ENV PATH="${PATH}:/opt/hpcx/ompi/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/hpcx/ompi/lib"
ENV PATH="${PATH}:/opt/hpcx/ucx/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/hpcx/ucx/lib"


RUN pip install "Cython<3.0" "pyyaml<6" --no-build-isolation

ARG USE_CUDA=0
ARG TORCH_ARCH="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
ENV AM_I_DOCKER True
ENV BUILD_WITH_CUDA "${USE_CUDA}"
ENV TORCH_CUDA_ARCH_LIST "${TORCH_ARCH}"
ENV CUDA_HOME /usr/local/cuda-11.6/

RUN bash ./install_requirements.sh
RUN pip install -e .
RUN pip install torchtext==0.13.1

CMD ["bash"]

