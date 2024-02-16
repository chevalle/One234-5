#!/bin/bash


# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo apt-get install -y cuda-toolkit
sudo apt-get install -y nvidia-gds
sudo apt-get install -y cuda
sudo apt-get install -y nvidia-container-toolkit
aws ecr-public get-login-password --region us-east-1 | sudo docker login --username AWS --password-stdin public.ecr.aws
sudo docker pull public.ecr.aws/f8j2d1v7/one2345-3dprint:latest

export PATH=/usr/local/cuda/bin${PATH:+:${PATH}} >> ~/.bashrc 
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}} >> ~/.bashrc 
                         
sudo reboot now 

echo "Package installation script completed."


