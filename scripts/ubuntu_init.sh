wget https://repo.anaconda.com/archive/Anaconda3-2024.02-1-Linux-x86_64.sh
chmod +x Anaconda3-2024.02-1-Linux-x86_64.sh
./Anaconda3-2024.02-1-Linux-x86_64.sh
sudo apt-get update && sudo apt-get install libgl1
sudo apt install alsa-utils
sudo apt install ubuntu-drivers-common
sudo ubuntu-drivers autoinstall
sudo apt install unzip
sudo apt install nvidia-cuda-toolkit
sudo apt install nvidia-driver-545
sudo ubuntu-drivers install nvidia:545