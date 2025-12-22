# Dockerfile for unitree_rl_gym
# Based on the setup requirements from doc/setup_en.md

FROM nvcr.io/nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/opt/isaacgym/python:/opt/rsl_rl:$PYTHONPATH

# Install Python 3.8
RUN apt update && apt install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt update && \
    apt install -y python3.8 python3.8-dev python3.8-distutils && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

RUN apt update && apt install -y curl

# Install pip for Python 3.8
RUN curl https://bootstrap.pypa.io/pip/3.8/get-pip.py -o get-pip.py && \
    python3.8 get-pip.py && \
    rm get-pip.py

RUN pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121

COPY ./isaacgym /opt/isaacgym
RUN cd /opt/isaacgym/python && \
    pip install -e .

RUN apt update && apt install -y git

# Clone and install rsl_rl
WORKDIR /opt
RUN git clone https://github.com/leggedrobotics/rsl_rl.git && \
    cd rsl_rl && \
    git checkout v1.0.2 && \
    pip install -e .

# Clone and install unitree_sdk2py (optional)
RUN git clone https://github.com/unitreerobotics/unitree_sdk2_python.git && \
    cd unitree_sdk2_python && \
    pip install -e .

# Copy the current project
COPY . /opt/unitree_rl_gym
WORKDIR /opt/unitree_rl_gym

# Install unitree_rl_gym
RUN pip install -e .

# Set working directory
WORKDIR /opt/unitree_rl_gym

# Set the default command
CMD ["/bin/bash"]
