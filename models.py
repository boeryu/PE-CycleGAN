import torch.nn as nn
import torch.nn.functional as F
from torch.nn import InstanceNorm3d
from torch.nn.functional import max_pool3d


class ResidualBlock3D(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        layers = [
            nn.ReflectionPad3d(1),
            nn.Conv3d(in_features, in_features, 3),
            InstanceNorm3d(in_features),
            nn.ReLU(inplace=True)
        ]

        layers += [
            nn.ReflectionPad3d(1),
            nn.Conv3d(in_features, in_features, 3),
            InstanceNorm3d(in_features),
        ]
        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator3D(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=6):
        super(Generator3D, self).__init__()

        # Initial convolution block
        model = [nn.ReflectionPad3d(3),
                 nn.Conv3d(input_nc, 32, 7),
                 InstanceNorm3d(32),
                 nn.ReLU(inplace=True)]

        # Downsampling
        in_features = 32
        out_features = in_features * 2
        for _ in range(3):
            model += [

                nn.Conv3d(in_features, out_features, kernel_size=3, stride=1, padding=1),
                nn.MaxPool3d(2),
                nn.Conv3d(out_features, out_features, 3, 1, 1),
                InstanceNorm3d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features *= 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock3D(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(3):
            model += [
                nn.Conv3d(in_features, in_features, 3, 1, 1),
                InterpolateUpsample3D(scale_factor=2, mode='trilinear', align_corners=True),
                nn.Conv3d(in_features,out_features,3,1,1),
                InstanceNorm3d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features //= 2
        model += [
            nn.ReflectionPad3d(3),
            nn.Conv3d(in_features, output_nc, kernel_size=7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):

        return self.model(x)


class InterpolateUpsample3D(nn.Module):
    def __init__(self, scale_factor, mode='trilinear', align_corners=True):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        return F.interpolate(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners
        )

class Discriminator3D(nn.Module):
    def __init__(self, input_nc):
        super().__init__()

        self.model = nn.Sequential(

            nn.Conv3d(input_nc, 64, kernel_size=4, stride=2, padding=(1, 1, 1)),
            InstanceNorm3d(64),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1),
            InstanceNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1),
            InstanceNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1),
            InstanceNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(512, 1, kernel_size=1),
            nn.Flatten()
        )

    def forward(self, x):
        return self.model(x)
