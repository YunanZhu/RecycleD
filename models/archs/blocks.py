from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as nn_f

from models.archs.arch_utils import *


class Identity(nn.Module):
    def __init__(self):
        """
        A block which will directly return the input tensor.
        """
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class ResidualBlock(nn.Module):
    def __init__(self,
                 channels: int = 64,
                 res_scale: Union[float, int] = 1.0,
                 esrgan_init: bool = False):
        """
        Residual Block with BN layers.

        :param channels:
            The number of input/output/intermediate feature channels.
        :param res_scale:
            A real number to rescale the residual.
        :param esrgan_init:
            Use ESRGAN initialization if set to True, otherwise use Pytorch default initialization.
        """
        super(ResidualBlock, self).__init__()

        assert isinstance(res_scale, (float, int)), (
            f"The res_scale should be a real number, but get {type(res_scale)}."
        )
        self.res_scale = float(res_scale)

        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=False),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(channels),
        )

        if esrgan_init:
            default_init_weights(self.conv_block, 0.1)

    def forward(self, x):
        return x + self.conv_block(x) * self.res_scale


class ResidualBlockNoBN(nn.Module):
    def __init__(self,
                 channels: int = 64,
                 res_scale: Union[float, int] = 1.0,
                 esrgan_init: bool = False):
        """
        Residual Block w/o BN layer.
        It use LeakyReLU as activation instead of ReLU.

        :param channels:
            The number of input/output/intermediate feature channels.
        :param res_scale:
            A real number to rescale the residual.
        :param esrgan_init:
            Use ESRGAN initialization if set to True, otherwise use Pytorch default initialization.
        """
        super(ResidualBlockNoBN, self).__init__()

        assert isinstance(res_scale, (float, int)), (
            f"The res_scale should be a real number, but get {type(res_scale)}."
        )
        self.res_scale = float(res_scale)

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        if esrgan_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        residual = self.conv2(self.lrelu(self.conv1(x)))
        return x + residual * self.res_scale


class DenseBlock(nn.Module):
    def __init__(self,
                 channels: int = 64,
                 grow_channels: int = 32,
                 esrgan_init: bool = True):
        """
        Dense Block.
        It is used in Residual Dense Block in RRDB in ESRGAN.

        :param channels:
            The number of input/output feature channels.
        :param grow_channels:
            The number of channels for each growth.
        :param esrgan_init:
            Use ESRGAN initialization if set to True, otherwise use Pytorch default initialization.
        """
        super(DenseBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, grow_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels + grow_channels, grow_channels, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(channels + 2 * grow_channels, grow_channels, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(channels + 3 * grow_channels, grow_channels, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(channels + 4 * grow_channels, channels, kernel_size=3, stride=1, padding=1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        if esrgan_init:
            default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), dim=1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), dim=1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), dim=1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), dim=1))
        return x5


class ResidualDenseBlock(nn.Module):
    def __init__(self,
                 channels: int = 64,
                 grow_channels: int = 32,
                 res_scale: Union[float, int] = 0.2,
                 esrgan_init: bool = True):
        """
        Residual Dense Block.
        It is used in Residual in Residual Dense Block (RRDB) in ESRGAN.

        :param channels:
            The number of input/output feature channels.
        :param grow_channels:
            The number of channels for each growth in Dense Block.
        :param res_scale:
            A real number to rescale the residual.
            Empirically use 0.2 to rescale the residual for better performance in ESRGAN.
        :param esrgan_init:
            Use ESRGAN initialization if set to True, otherwise use Pytorch default initialization.
        """
        super(ResidualDenseBlock, self).__init__()

        assert isinstance(res_scale, (float, int)), (
            f"The res_scale should be a real number, but get {type(res_scale)}."
        )
        self.res_scale = float(res_scale)

        self.dense_block = DenseBlock(
            channels=channels,
            grow_channels=grow_channels,
            esrgan_init=esrgan_init
        )

    def forward(self, x):
        return x + self.dense_block(x) * self.res_scale


class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self,
                 channels: int,
                 grow_channels: int = 32,
                 res_scale: Union[float, int] = 0.2,
                 esrgan_init: bool = True):
        """
        Residual in Residual Dense Block (RRDB).
        It is used in RRDBNet in ESRGAN.

        :param channels:
            The number of input/output feature channels.
        :param grow_channels:
            The number of channels for each growth in Residual Dense Block.
        :param res_scale:
            A real number to rescale the residual.
            Empirically use 0.2 to rescale the residual for better performance in ESRGAN.
        :param esrgan_init:
            Use ESRGAN initialization if set to True, otherwise use Pytorch default initialization.
        """
        super(ResidualInResidualDenseBlock, self).__init__()

        assert isinstance(res_scale, (float, int)), (
            f"The res_scale should be a real number, but get {type(res_scale)}."
        )
        self.res_scale = float(res_scale)

        # Stack several same RDBs.
        rdb_list = [
            ResidualDenseBlock(
                channels=channels,
                grow_channels=grow_channels,
                res_scale=res_scale,
                esrgan_init=esrgan_init
            ) for _ in range(3)
        ]
        self.rdb_stack = nn.Sequential(*rdb_list)

    def forward(self, x):
        return x + self.rdb_stack(x) * self.res_scale


class PixelShuffleUpsampleBlock(nn.Module):
    def __init__(self,
                 channels: int,
                 upscale_factor: int = 2):
        """
        Upsample by using PixelShuffle operation.

        :param channels:
            The number of input/output feature channels.
        :param upscale_factor:
            Upscale factor.
            Supported values: {2 ** n, 3}.
        """
        super(PixelShuffleUpsampleBlock, self).__init__()

        assert isinstance(upscale_factor, int), f"The upscale_factor type {type(upscale_factor)} is illegal."
        assert upscale_factor >= 2

        # Here satisfy "x & (x - 1) == 0" only when {x = 2 ** n} where n is positive integer.
        if (upscale_factor & (upscale_factor - 1)) == 0:
            import math
            block = []
            for _ in range(int(math.log(upscale_factor, 2) + 1e-8)):
                block.extend([
                    nn.Conv2d(channels, channels * 4, kernel_size=3, stride=1, padding=1),
                    nn.PixelShuffle(upscale_factor=2),
                    nn.LeakyReLU(negative_slope=0.1, inplace=True)
                ])
        elif upscale_factor == 3:
            block = [
                nn.Conv2d(channels, channels * 9, kernel_size=3, stride=1, padding=1, bias=True),
                nn.PixelShuffle(upscale_factor=3),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)
            ]
        else:
            raise NotImplementedError(
                f'The upscale_factor = {upscale_factor} is not supported.'
                f'The supported upscale factor values: 2^n and 3.'
            )

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class InterpolateUpsampleBlock(nn.Module):
    def __init__(self,
                 channels: int,
                 upscale_factor: int = 2,
                 mode: str = "nearest"):
        """
        Upsample by using interpolation.

        :param channels:
            The number of input/output feature channels.
        :param upscale_factor:
            Upscale factor.
            Supported values: {2 ** n, 3}.
        :param mode:
            The algorithm used for upsampling.
            Supported values: "nearest", "bilinear" and "bicubic".
            Default: "nearest".
            We set align_corners=False according to xinntao's BasicSR.
        """
        super(InterpolateUpsampleBlock, self).__init__()

        assert isinstance(upscale_factor, int), f"The upscale_factor type {type(upscale_factor)} is illegal."
        assert upscale_factor >= 2

        assert isinstance(mode, str) and mode in ("nearest", "bilinear", "bicubic"), f"Illegal mode: {mode}."

        # Here satisfy "x & (x - 1) == 0" only when {x = 2 ** n} where n is positive integer.
        if (upscale_factor & (upscale_factor - 1)) == 0:
            import math
            block = []
            for _ in range(int(math.log2(upscale_factor) + 1e-8)):
                block.extend([
                    nn.Upsample(scale_factor=2, mode=mode),
                    nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True)
                ])
        elif upscale_factor == 3:
            block = [
                nn.Upsample(scale_factor=3, mode=mode),
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ]
        else:
            raise NotImplementedError(
                f'The upscale_factor = {upscale_factor} is not supported.'
                f'The supported upscale factor values: 2^n and 3.'
            )

        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


if __name__ == '__main__':
    pass
