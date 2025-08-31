# PE-CycleGAN
本项目用于计算超分辨率显微图像，有两个文件夹，包含了Python文件和Matlab文件，环境配置是Conda环境
# Conda环境配置
torch2.4_cuda12.1
# 数据集
使用了[Bio-LFSR](https://zenodo.org/records/7233421)数据集，读者请自行下载
超分辨率与低分辨率图像数据[重建代码](https://github.com/THU-IBCS/VsLFM-master)是

# CycleGAN
基于精简的[CycleGAN](https://github.com/aitorzip/PyTorch-CycleGAN.git)进行了改动
## Training
#### Datasets

在数据集里放入所需的超分辨率图像与低分辨率图像(A域代表SR图像，B域代表LR图像)

```
├── datasets                   
|   ├── <dataset_name>        
|   |   ├── train              # Training
|   |   |   ├── A              # Contains domain A images
|   |   |   └── B              # Contains domain B images
|   |   └── test               # Testing
|   |   |   ├── A              # Contains domain A images
|   |   |   └── B              # Contains domain B images
```

打开train文件将训练参数设置为

```
./train 
--dataroot datasets/<dataset_name>/ 
--cuda
```

### ### Testing

打开test文件将参数设置为

```
./test 
--dataroot datasets/<dataset_name>/ 
--cuda
```

## PE-CycleGAN

该文件夹内容是基于物理模型嵌入的CycleGAN,使用方法与CycleGAN一致
