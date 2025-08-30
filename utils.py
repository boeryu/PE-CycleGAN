import random
import time
import datetime
import sys
import torch.nn.functional as F
from PIL.ImageOps import scale
from networkx.algorithms.cuts import volume
from torch.autograd import Variable
import torch
import torch.nn as nn
from visdom import Visdom
import numpy as np
from torchvision import transforms

class Logger3D():
    def __init__(self, n_epochs, batches_epoch):
        self.viz = Visdom()
        self.n_epochs = n_epochs
        self.batches_epoch = batches_epoch
        self.epoch = 1
        self.batch = 1
        self.prev_time = time.time()
        self.mean_period = 0
        self.losses = {}
        self.loss_windows = {}
        self.image_windows = {}

    def log(self, losses=None, images=None):
        self.mean_period += (time.time() - self.prev_time)
        self.prev_time = time.time()

        sys.stdout.write(
            '\rEpoch %03d/%03d [%04d/%04d] -- ' % (self.epoch, self.n_epochs, self.batch, self.batches_epoch))

        for i, loss_name in enumerate(losses.keys()):
            loss_value = losses[loss_name].item()
            if loss_name not in self.losses:
                self.losses[loss_name] = loss_value
            else:
                self.losses[loss_name] += loss_value

            if (i + 1) == len(losses.keys()):
                sys.stdout.write('%s: %.4f -- ' % (loss_name, self.losses[loss_name] / self.batch))
            else:
                sys.stdout.write('%s: %.4f | ' % (loss_name, self.losses[loss_name] / self.batch))

        batches_done = self.batches_epoch * (self.epoch - 1) + self.batch
        batches_left = self.batches_epoch * (self.n_epochs - self.epoch) + self.batches_epoch - self.batch
        sys.stdout.write('ETA: %s' % (datetime.timedelta(seconds=batches_left * self.mean_period / batches_done)))

        if (self.batch % self.batches_epoch) == 0:
            # Plot losses
            for loss_name, loss in self.losses.items():
                if loss_name not in self.loss_windows:
                    self.loss_windows[loss_name] = self.viz.line(X=np.array([self.epoch]),
                                                                 Y=np.array([loss / self.batch]),
                                                                 opts={'xlabel': 'epochs', 'ylabel': loss_name,
                                                                       'title': loss_name})
                else:
                    self.viz.line(X=np.array([self.epoch]), Y=np.array([loss / self.batch]),
                                  win=self.loss_windows[loss_name], update='append')
                # Reset losses for next epoch
                self.losses[loss_name] = 0.0

            self.epoch += 1
            self.batch = 1
            sys.stdout.write('\n')
        else:
            self.batch += 1


class ReplayBuffer3D():
    def __init__(self, max_size=50):
        assert (max_size > 0), 'Empty buffer or trying to create a black hole. Be careful.'
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[idx].clone())
                    self.data[idx] = element
                else:
                    to_return.append(element)
        return torch.cat(to_return)


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)





class RandomAffine3D:
    """修正后的3D仿射变换类"""

    def __init__(self, degrees=10, translate=0.1, scale=(0.9, 1.1)):
        self.degrees = degrees if isinstance(degrees, (tuple, list)) else (degrees, degrees, degrees)
        self.translate = translate
        self.scale = scale if isinstance(scale, tuple) else (max(0.1, 1 - scale), 1 + scale)

    def __call__(self, volume):
        # volume shape: [C, D, H, W]

        has_batch = (volume.dim() == 5)
        if not has_batch:
            volume = volume.unsqueeze(0)
        B, C, D, H, W = volume.shape

        # 生成随机变换参数
        angle_z = np.random.uniform(-self.degrees[0], self.degrees[0])
        angle_y = np.random.uniform(-self.degrees[1], self.degrees[1])
        angle_x = np.random.uniform(-self.degrees[2], self.degrees[2])

        max_dz = self.translate * D
        max_dy = self.translate * H
        max_dx = self.translate * W
        trans_z = np.random.uniform(-max_dz, max_dz)
        trans_y = np.random.uniform(-max_dy, max_dy)
        trans_x = np.random.uniform(-max_dx, max_dx)

        scale_x = np.random.uniform(*self.scale)
        scale_y = np.random.uniform(*self.scale)
        scale_z = np.random.uniform(*self.scale)
        # 生成3D仿射矩阵
        affine_matrix = self._get_affine_matrix(
            angles=(angle_x, angle_y, angle_z),
            translate=(trans_x, trans_y, trans_z),
            scale=(scale_x, scale_y, scale_z),
            img_size=(B, C, D, H, W)
        )

        # 应用3D仿射变换
        grid = F.affine_grid(affine_matrix, volume.size(), align_corners=False)
        transformed = F.grid_sample(volume, grid, mode='bilinear', padding_mode='reflection', align_corners=False)
        if not has_batch:
            transformed = transformed.squeeze(0)
        return transformed

    def _get_affine_matrix(self, angles, translate, scale, img_size):
        """生成3D仿射矩阵"""
        scale_x, scale_y, scale_z = scale
        B, C, D, H, W = img_size
        angle_x, angle_y, angle_z = angles
        tx, ty, tz = translate
        scale_mat = torch.tensor([
            [scale_x, 0, 0, 0],
            [0, scale_y, 0, 0],
            [0, 0, scale_z, 0],
            [0, 0, 0, 1]
        ], dtype=torch.float32)
        # 将角度转换为弧度
        theta_x = torch.deg2rad(torch.tensor(angles[0]))
        theta_y = torch.deg2rad(torch.tensor(angles[1]))
        theta_z = torch.deg2rad(torch.tensor(angles[2]))

        # 构建旋转矩阵
        Rz = torch.tensor([
            [torch.cos(theta_z), -torch.sin(theta_z), 0, 0],
            [torch.sin(theta_z), torch.cos(theta_z), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        Ry = torch.tensor([
            [torch.cos(theta_y), 0, torch.sin(theta_y), 0],
            [0, 1, 0, 0],
            [-torch.sin(theta_y), 0, torch.cos(theta_y), 0],
            [0, 0, 0, 1]
        ])

        Rx = torch.tensor([
            [1, 0, 0, 0],
            [0, torch.cos(theta_x), -torch.sin(theta_x), 0],
            [0, torch.sin(theta_x), torch.cos(theta_x), 0],
            [0, 0, 0, 1]
        ])

        # 组合旋转
        rotation = Rx @ Ry @ Rz
        tx_norm = 2 * tx / (W - 1)
        ty_norm = 2 * ty / (H - 1)
        tz_norm = 2 * tz / (D - 1)
        trans_mat = torch.tensor([
            [1, 0, 0, tx_norm],
            [0, 1, 0, ty_norm],
            [0, 0, 1, tz_norm],
            [0, 0, 0, 1]
        ])

        affine_matrix = trans_mat @ rotation @ scale_mat
        return affine_matrix[:3, :].unsqueeze(0).repeat(B, 1, 1).float()  # 返回形状 [1, 3, 4]


class RandomHorizontalFlip3D:
    """3D水平翻转"""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, volume):
        if torch.rand(1) < self.p:
            return volume.flip(-1)  # 翻转宽度维度
        return volume


class RandomCrop3D:
    """3D随机裁剪"""

    def __init__(self, output_size):
        self.output_size = output_size  # (D, H, W)

    def __call__(self, volume):
        has_batch = (volume.dim() == 5)
        if not has_batch:
            volume = volume.unsqueeze(0)
        _, c, d, h, w = volume.shape
        new_d, new_h, new_w = self.output_size

        new_d = min(d, new_d)
        new_h = min(h, new_h)
        new_w = min(w, new_w)

        d_start = torch.randint(0, max(1, d - new_d), (1,))
        h_start = torch.randint(0, max(1, h - new_h), (1,))
        w_start = torch.randint(0, max(1, w - new_w), (1,))

        return volume[
               :,
               :,
               d_start:d_start + new_d,
               h_start:h_start + new_h,
               w_start:w_start + new_w
               ]
def weights_init_3d(m):
    """修正后的3D网络初始化函数"""
    if isinstance(m, nn.Conv3d):
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.trunc_normal_(m.weight.data, mean=0.0, std=0.02)
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, (nn.InstanceNorm3d, nn.BatchNorm3d)):
        if hasattr(m, "weight") and m.weight is not None:
            nn.init.constant_(m.weight.data, 1.0)  # 归一化层gamma初始化
        if hasattr(m, "bias") and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)    # 归一化层beta初始化

class FixedDepthCrop3D:
    """3D固定深度裁剪 (深度从首层开始，高度宽度保持原样)"""

    def __init__(self, output_size):
        self.output_size = output_size  # (D, H, W)

    def __call__(self, volume):
        has_batch = (volume.dim() == 5)
        if not has_batch:
            volume = volume.unsqueeze(0)
        _, c, d, h, w = volume.shape
        new_d, new_h, new_w = self.output_size

        new_d = min(d, new_d)
        new_h = min(h, new_h)
        new_w = min(w, new_w)

        d_start = 0
        h_start = h // 2 - new_h // 2
        w_start = w // 2 - new_w // 2

        return volume[
               :,
               :,
               d_start:d_start + new_d,
               h_start:h_start + new_h,
               w_start:w_start + new_w
               ]




