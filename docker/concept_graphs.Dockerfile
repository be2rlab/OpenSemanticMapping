FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel AS base


ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && apt install -y git-all


# Install Python dependencies

RUN apt-get update -y && apt-get install -y python3-tk
WORKDIR /opt/src

ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
ENV PYTHONPATH "${PYTHONPATH}:/opt/src"
ENV PATH="${PATH}:/opt/hpcx/ompi/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/hpcx/ompi/lib"
ENV PATH="${PATH}:/opt/hpcx/ucx/bin"
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/opt/hpcx/ucx/lib"

RUN conda update -n base -c defaults conda
RUN conda install -c pytorch faiss-cpu=1.7.4 mkl=2021 blas=1.0=mkl

RUN conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y
RUN conda install https://anaconda.org/pytorch3d/pytorch3d/0.7.5/download/linux-64/pytorch3d-0.7.5-py310_cu118_pyt210.tar.bz2

RUN pip install --upgrade pip

RUN pip install tyro open_clip_torch wandb h5py openai hydra-core

COPY ${ROOT_DIR}/Thirdparty /tmp/
# # Install the gradslam package and its dependencies
# RUN git clone https://github.com/JaafarMahmoud1/chamferdist.git \
RUN cd /tmp/chamferdist \
&& pip install . 
# && git clone https://github.com/gradslam/gradslam.git \
RUN cd /tmp/gradslam \
# && git checkout conceptfusion \
&& pip install . 

FROM base AS dev
ARG USE_CUDA=0
ARG TORCH_ARCH=
ENV AM_I_DOCKER True
ENV BUILD_WITH_CUDA "${USE_CUDA}"
ENV TORCH_CUDA_ARCH_LIST "${TORCH_ARCH}"
ENV CUDA_HOME /usr/local/cuda-11.8/

WORKDIR /tmp/Grounded-Segment-Anything

RUN apt-get update -y && apt-get install --no-install-recommends wget ffmpeg=7:* \
    libsm6=2:* libxext6=2:* -y \
    && apt-get clean && apt-get autoremove && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir -e segment_anything && \
    python -m pip install --no-cache-dir -e GroundingDINO

RUN pip install --no-cache-dir diffusers[torch]==0.15.1 opencv-python==4.7.0.72 \
    pycocotools==2.0.6 matplotlib==3.5.3 \
    onnxruntime==1.14.1 onnx==1.13.1 ipykernel==6.16.2 scipy gradio openai

RUN python3 -c "import open_clip; \
                open_clip.create_model_and_transforms(\"ViT-H-14\", \"laion2b_s32b_b79k\");"

WORKDIR /opt/src
COPY ${ROOT_DIR}/concept-graphs ./

COPY ${ROOT_DIR}/concept-graphs/requirements.txt requirements.txt
RUN pip install -r requirements.txt

WORKDIR /opt/src
RUN pip install .

CMD ["bash"]
