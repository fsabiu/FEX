conda config --add channels conda-forge
conda install imgaug
conda create --name imgaug python=3.8 -y
conda activate imgaug
pip install imgaug