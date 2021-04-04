#!/bin/sh

# Linux Ubuntu 20.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.2.1/local_installers/cuda-repo-ubuntu2004-11-2-local_11.2.1-460.32.03-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-2-local_11.2.1-460.32.03-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-2-local/7fa2af80.pub
sudo apt update
sudo apt install cuda -y

# Cuda Tool Kit
sudo apt install nvidia-cuda-toolkit -y

# Export Path
#echo "# Cuda" >> /home/us3r/.bashrc
#echo "export PATH=/usr/local/cuda/bin:$PATH" >> /home/us3r/.bashrc
#echo "export CPATH=/usr/local/cuda/include:$CPATH" >> /home/us3r/.bashrc
#echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH" >> /home/us3r/.bashrc
#echo "export DYLD_LIBRARY_PATH=/usr/local/cuda/lib:$DYLD_LIBRARY_PATH" >> /home/us3r/.bashrc

# pip command
# pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
