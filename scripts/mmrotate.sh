# execute one by one manually
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c nvidia
pip install -U openmim
mim install mmcv-full
mim install mmdet\<3.0.0
pip install mmrotate
git clone https://github.com/open-mmlab/mmrotate.git
sudo apt-get update && sudo apt-get install libgl1
sudo apt install ubuntu-drivers-common
sudo ubuntu-drivers autoinstall