FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel AS base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt install -y git-all

WORKDIR /opt/src

COPY ${ROOT_DIR}/openscene ./

ENV TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"
ENV CUDA_HOME="/usr/local/cuda-11.6"
ENV CXX="c++"

RUN apt install -y build-essential python3-dev libopenblas-dev

RUN pip install ninja==1.10.2.3

RUN conda install -y openblas-devel -c anaconda

RUN pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps --install-option="--force_cuda"

RUN pip install -r requirements.txt
RUN pip install tensorflow

### Prepare for visualization
RUN apt install unzip -y
RUN apt install wget -y 
RUN apt install libglew-dev -y

WORKDIR /opt/src/demo/gaps
RUN make
WORKDIR /opt/src/demo/gaps/pkgs/RNNets
RUN make
WORKDIR /opt/src/demo/gaps/apps/osview
RUN make

WORKDIR /opt/src

CMD ["bash"]
