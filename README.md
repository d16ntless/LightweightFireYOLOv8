# Lightweight-YOLOv8-for-Real-Time-Forest-Fire-Detection-via-Inverted-Residual-and-Biformer-Attention

dataset：
https://drive.google.com/file/d/11YwVAph_-b8Ew25zM-MYGQO-TqIbu-zE/view?usp=drive_link

environment:
python: 3.8.16
torch: 1.13.1+cu117
torchvision: 0.14.1+cu117
timm: 0.9.8
mmcv: 2.1.0
mmengine: 0.9.0
pip: pip install timm==0.9.8 thop efficientnet_pytorch==0.7.1 einops grad-cam==1.4.8 dill==0.3.6 albumentations==1.3.1 pytorch_wavelets==1.3.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

Script of training model： train.py
Script for calculating indicators using the trained model： val.py
