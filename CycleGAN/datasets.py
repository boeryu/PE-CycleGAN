import glob
import random
import os
import tifffile
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np

class VolumeDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=False, mode='test'):
        self.mode = mode
        self.unaligned = unaligned
        self.files_A = sorted(glob.glob(os.path.join(root, f'{mode}/A/*.tif')))
        self.files_B = sorted(glob.glob(os.path.join(root, f'{mode}/B/*.tif')))

        default_transforms=[
            transforms.Lambda(lambda x: self._ensure_4d(x)),
        ]

        self.transform = transforms.Compose(
            transforms_ if transforms_ else default_transforms
        )
    def __getitem__(self, index):
        vol_A = tifffile.imread(self.files_A[index % len(self.files_A)])
        vol_A = torch.from_numpy(vol_A.astype(np.float32))

        if self.unaligned:
            vol_B=tifffile.imread(random.choice(self.files_B))
        else:
             vol_B = tifffile.imread(self.files_B[index % len(self.files_B)])
        vol_B = torch.from_numpy(vol_B.astype(np.float32))


        vol_A = self.transform(vol_A)
        vol_B = self.transform(vol_B)

        return {'A': vol_A, 'B': vol_B}


    def __len__(self):
        return max(len(self.files_A), len(self.files_B))
    @staticmethod
    def _ensure_4d(x):
        if x.dim()==3:
            return x.unsqueeze(0)
        elif x.dim()==4:
            return x
