# PE-CycleGAN
本项目用于计算超分辨率显微图像，有两个文件夹，包含了Python文件和Matlab文件，环境配置是Conda环境
# Conda环境配置
torch2.4_cuda12.1
# 数据集
使用了Bio-LFSR数据集，读者请自行前往 https://zenodo.org/records/7233421下载
超分辨率与低分辨率图像数据重建代码来自 https://github.com/THU-IBCS/VsLFM-master
# CycleGAN
基于精简的CycleGAN(https://github.com/aitorzip/PyTorch-CycleGAN.git)进行了改动
-
# Training
1.在数据集里放入所需的超分辨率图像与低分辨率图像
.
├── datasets                   
|   ├── <dataset_name>         # i.e. brucewayne2batman
|   |   ├── train              # Training
|   |   |   ├── A              # Contains domain A images (i.e. Bruce Wayne)
|   |   |   └── B              # Contains domain B images (i.e. Batman)
|   |   └── test               # Testing
|   |   |   ├── A              # Contains domain A images (i.e. Bruce Wayne)
|   |   |   └── B              # Contains domain B images (i.e. Batman)
