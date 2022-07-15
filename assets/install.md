# Install
## Requirements
We test the codes in the following environments, other versions may also be compatible but Pytorch vision should be >= 1.7

- CUDA 11.3
- Python 3.7
- Pytorch 1.10.0
- Torchvison 0.11.1

## Install environment for Unicorn

```
# Pytorch and Torchvision
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge

# YOLOX and some other packages
pip3 install -U pip && pip3 install -r requirements.txt
pip3 install -v -e .  # or python3 setup.py develop

# Install Deformable Attention
cd unicorn/models/ops
bash make.sh
cd ../../..

# Install mmcv, mmdet, bdd100k
cd external/qdtrack
wget -c https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/mmcv_full-1.4.6-cp37-cp37m-manylinux1_x86_64.whl # This should change according to cuda version and pytorch version
pip3 install --user mmcv_full-1.4.6-cp37-cp37m-manylinux1_x86_64.whl
pip3 install --user mmdet
git clone https://github.com/bdd100k/bdd100k.git
cd bdd100k
python3 setup.py develop --user
pip3 uninstall -y scalabel
pip3 install --user git+https://github.com/scalabel/scalabel.git
cd ../../..
```
