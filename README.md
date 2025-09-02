# Physics Embedded CycleGAN(PE-CycleGAN)--More efficient and stable

本项目基于CycleGAN模型，在模型中嵌入点扩散函数替代深度学习生成器实现了更简单的由超分辨率到低分辨率的转化，降低了训练时计算成本，并且以高效、稳定的方式实现了计算超分辨率显微图像。该项目中有两个文件夹，分别是三维的CycleGAN模型代码和我们的PE-CycleGAN代码，文件夹中包含了Python文件和Matlab文件两种文件类型，文件运行环境配置是Conda(torch2.4_cuda12.1)

## 数据集
训练与测试的数据使用了由清华大学成像与智能技术实验室戴琼海院士团队发布的国际首个大规模多样本的显微光场超分辨率数据集[Bio-LFSR](https://zenodo.org/records/7233421)数据集，其中包含了训练与测试所用的超分辨率与低分辨率样本图像，重建图像的代码来自于其团队开发的[VsLFM-master](https://github.com/THU-IBCS/VsLFM-master)

## CycleGAN
基于精简的二维[CycleGAN](https://github.com/aitorzip/PyTorch-CycleGAN.git)模型，我们将其改成适用于三维图像的模型。
### 1.Training
在数据集里放入训练所需的超分辨率图像与低分辨率图像(例：A域代表SR图像，B域代表LR图像)，我们将图片顺序打乱进行无监督学习

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
--dataroot 
datasets/<dataset_name>
--cuda
```

### 2.Testing

打开test文件将参数设置为，运行文件可以将测试集中的超分辨率图像转换为低分辨率图像，低分辨率图像转换为超分辨率图像，并保存到指定的文件中

```
--dataroot
datasets/<datasets name>
--generator_B2A
output/<datasets name>/netG_B2A.pth
--generator_A2B
output/<datasets name>/netG_B2A.pth
--cuda
```

## PE-CycleGAN

该文件夹内容是物理模型嵌入的CycleGAN，我们仅训练了由低分辨率图像转换为超分辨率图像的模型。在测试中，我们也仅获得由低分辨率图像计算出的超分辨率图像；用于嵌入模型的高斯核由gaussian.m文件生成。

## Results!

我们准备了两个模型分别用15组超、低分辨率图像训练出的模型与测试用的图像，并上传至百度网盘，请读者自行[下载](https://pan.baidu.com/s/1Ev7ou1Ew5eNQn58ikqgcKQ?pwd=ahjw )并运行代码查看结果

## 致谢

感谢项目的贡献者开源

## 联系

如果你有任何问题或建议，请随时通过我的电子邮件（1691128029@qq.com）与我联系
