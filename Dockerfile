FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-devel


# Set up time zone.
ENV TZ=UTC
ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"
ENV TORCH_CUDA_ARCH_LIST="8.6+PTX"
ENV IABN_FORCE_CUDA=1
ENV PATH="/usr/local/cuda-11.8/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH"
ENV CUDA_HOME='/usr/local/cuda-11.8'
RUN which nvcc
WORKDIR /code

# Install system libraries required by OpenCV.
RUN apt-get update \
 && apt-get install -y libgl1-mesa-glx libgtk2.0-0 libsm6 libxext6 \
  build-essential \
  python3.9 \
  python3-pip \
  git \
  ffmpeg \
  wget \
  libsparsehash-dev \
  gcc-11 \
  g++-11 \
  libsm6 libxrender1 libfontconfig1 libxext6 \
  libxrender-dev \
  nginx \
  ca-certificates \
 && rm -rf /var/lib/apt/lists/*


RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py && \
    pip install flask gevent gunicorn && \
        rm -rf /root/.cache


COPY ./requirements.txt /code/requirements.txt
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 60 --slave /usr/bin/g++ g++ /usr/bin/g++-11
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install -r requirements.txt
RUN TORCH_CUDA_ARCH_LIST="8.6+PTX" IABN_FORCE_CUDA=1 FORCE_CUDA=1 pip3 install --no-cache-dir inplace_abn
RUN FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.6+PTX" IABN_FORCE_CUDA=1 pip install --no-cache-dir git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0

WORKDIR /opt/program

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY . /opt/program

#CMD ["python3", "serve.py"]
