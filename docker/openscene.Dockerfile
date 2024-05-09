FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel AS base

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt install -y git-all

WORKDIR /opt/src/
COPY ${ROOT_DIR}/openscene ./

RUN apt-get install -y libopenexr-dev

RUN pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
RUN apt install -y build-essential python3-dev libopenblas-dev


RUN pip install -U git+https://github.com/NVIDIA/MinkowskiEngine -v --no-deps \
--config-settings="--cuda=force_cuda" \
--config-settings="--blas=openblas"

RUN pip install -r requirements.txt

RUN pip install tensorflow

CMD ["bash"]
