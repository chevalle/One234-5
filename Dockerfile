FROM pytorch/pytorch:2.2.1-cuda11.8-cudnn8-devel


# Set up time zone.
ENV TZ=UTC
ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
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
 && rm -rf /var/lib/apt/lists/*


COPY ./requirements.txt /code/requirements.txt
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 60 --slave /usr/bin/g++ g++ /usr/bin/g++-11
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install -r requirements.txt
RUN TORCH_CUDA_ARCH_LIST="8.6+PTX" IABN_FORCE_CUDA=1 FORCE_CUDA=1 pip3 install --no-cache-dir inplace_abn
RUN FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.6+PTX" IABN_FORCE_CUDA=1 pip install --no-cache-dir git+https://github.com/mit-han-lab/torchsparse.git@v1.4.0




# Set up a new user named "user" with user ID 1000
RUN useradd -m -u 1000 user
# Switch to the "user" user
USER user
# Set home to the user's home directory
ENV HOME=/home/user \
        PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=$HOME/app \
        PYTHONUNBUFFERED=1 \
        SYSTEM=spaces




# Set the working directory to the user's home directory
WORKDIR $HOME/app

# Copy the current directory contents into the container at $HOME/app setting the owner to the user
COPY --chown=user . $HOME/app

RUN python3 download_ckpt.py

CMD ["python3", "app.py"]
