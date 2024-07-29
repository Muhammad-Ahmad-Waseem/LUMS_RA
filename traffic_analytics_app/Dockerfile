FROM nvidia/cuda:11.0.3-cudnn8-runtime-ubuntu18.04
COPY . /opt/
WORKDIR /opt/
RUN apt-get update && apt-get install -y python3-opencv
RUN apt update && \
    apt install --no-install-recommends -y build-essential software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt install --no-install-recommends -y python3.8 python3-pip python3-setuptools python3-distutils && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN python3.8 -m pip install --upgrade pip && \
    python3.8 -m pip install --no-cache-dir -r requirements.txt && \
    python3.8 -m pip install --no-cache-dir torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
ENTRYPOINT python3.8 main.py
