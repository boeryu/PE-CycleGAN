#!/usr/bin/python3

import argparse
import itertools
import time
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import h5py
import numpy as np
import torch

from models import Generator3D,Discriminator3D


from utils import LambdaLR
from utils import Logger3D
from utils import weights_init_3d
from utils import RandomAffine3D
from utils import RandomCrop3D
from utils import ReplayBuffer3D
from utils import RandomHorizontalFlip3D
from datasets import VolumeDataset

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=501, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, required=True,default='datasets/l2r/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.00005, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=50, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--volume_shape', type=int,nargs=3, default=[16,256,256], help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--gpu_id', type=int, default=1, help='number of cpu threads to use during batch generation')
parser.add_argument('--dropout', type=bool, default=False, help='')
opt = parser.parse_args()
print(opt)

device = torch.device(f'cuda:{opt.gpu_id}' if opt.cuda and torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_name())
def load_mat_v73(path, key):
    """读取v7.3格式的.mat文件"""
    with h5py.File(path, 'r') as f:
        data = np.array(f[key]).transpose()  # 转置维度
    return data.astype(np.float32)  # 统一浮点类型


# 加载核数据
fixed_kernel = load_mat_v73(r'/data/OT-CycleGAN-master-3D/gausskernel17.mat', 'gauss3d')
###### Definition of variables ######
class MedicalAdaptiveBlur3D(torch.nn.Module):
   def __init__(self, kernel):
        super().__init__()
        self.conv3d = torch.nn.Conv3d(1, 1, kernel_size=kernel.shape, padding='same')
        with torch.no_grad():
            self.conv3d.weight.data = torch.from_numpy(kernel).unsqueeze(0).unsqueeze(0)
            self.conv3d.bias.data.zero_()
   def forward(self, x):
        return self.conv3d(x)
# Networks
netG_A2B = MedicalAdaptiveBlur3D(fixed_kernel).to(device)
netG_B2A = Generator3D(opt.output_nc, opt.input_nc).to(device)
netD_A = Discriminator3D(opt.input_nc).to(device)


netG_B2A.apply(weights_init_3d)
netD_A.apply(weights_init_3d)


# Lossess
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam( netG_B2A.parameters(),
                                lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))


lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)


def _normalized_3d(volume):
    esp = 1e-6
    return (volume-volume.min())/(volume.max()-volume.min()+esp)*2-1

train_transforms = [
    transforms.Lambda(lambda x:_auto_adjust_dims(x)),
    transforms.Lambda(lambda x:x.float()),
    RandomAffine3D(degrees=15,translate=0.1),
    RandomHorizontalFlip3D(p=0.5),
    RandomCrop3D(opt.volume_shape),
    transforms.Lambda(_normalized_3d)
]
def _auto_adjust_dims(tensor):
    original_dims=tensor.dim()
    if tensor.dim()==3:
        tensor = tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.dim()==4:
        tensor = tensor.unsqueeze(0)
    elif tensor.dim()==5:
      pass
    return tensor
dataset = VolumeDataset(
    root=opt.dataroot,
    transforms_=train_transforms,
    unaligned=True,
    mode='train'
)
dataloader = DataLoader(
    dataset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=opt.n_cpu,
    collate_fn=lambda x: x[0]
)

fake_A_buffer = ReplayBuffer3D()
fake_B_buffer = ReplayBuffer3D()

logger = Logger3D(opt.n_epochs, len(dataloader))
# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.FloatTensor

target_real = torch.ones(opt.batchSize,1,dtype=torch.float32,device=device)
target_fake = torch.zeros(opt.batchSize,1,dtype=torch.float32,device=device)

###################################
start_time=time.time()
###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):
        # Set model input

        real_A = batch['A'].to(device)
        real_B = batch['B'].to(device)

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()

        # Identity loss

        same_B = netG_A2B(real_B)
        loss_id_B = criterion_identity(same_B, real_B) * 5.0

        same_A = netG_B2A(real_A)
        loss_id_A = criterion_identity(same_A, real_A) * 5.0

        # GAN loss
        fake_B = netG_A2B(real_A)
        fake_A = netG_B2A(real_B)
        D_A_output = netD_A(fake_A)
        loss_GAN_B2A = criterion_GAN(D_A_output, target_real)

        # Cycle loss
        recovered_A = netG_B2A(fake_B)
        loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

        recovered_B = netG_A2B(fake_A)
        loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

        # Total loss
        loss_G =  loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB + loss_id_A + loss_id_B
        loss_G.backward()

        optimizer_G.step()
        ###################################

        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss
        pred_real = netD_A(real_A)
        loss_D_real = criterion_GAN(pred_real, target_real)

        # Fake loss
        fake_A = fake_A_buffer.push_and_pop(fake_A)
        pred_fake = netD_A(fake_A.detach())
        loss_D_fake = criterion_GAN(pred_fake, target_fake)

        # Total loss
        loss_D_A = (loss_D_real + loss_D_fake)*0.5
        loss_D_A.backward()

        optimizer_D_A.step()
        ###################################

        ###### Discriminator B ######

        ###################################
        # Progress report (http://localhost:8097)
        logger.log({'loss_G': loss_G,
                          'loss_id_A': loss_id_A ,
                          'loss_id_B':loss_id_B,
                          'loss_GAN':loss_GAN_B2A ,
                          'loss_cycle': loss_cycle_ABA+loss_cycle_BAB ,
                          'loss_D': loss_D_A },
                  images={'real_A': real_A,
                          'real_B': real_B,
                          'fake_A': fake_A,
                          'fake_B': fake_B
                            })

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()


    if epoch % 100==0:
    # Save models checkpoints
      torch.save(netG_B2A.state_dict(), f'output/ZGnoise/netG_B2A_{epoch}.pth')
      torch.save(netD_A.state_dict(), f'output/ZGnoise/netD_A_{epoch}.pth')

###################################
total_time=time.time()-start_time
print(f"Total training time:{total_time:.2f} seconds")
print("end successful")