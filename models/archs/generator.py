import torch
import torch.nn as nn
import torch.nn.functional as nn_f

from models.archs.blocks import *
from models.archs.arch_utils import *


class RRDBNet(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 n_feats: int = 64,
                 n_basic_blocks: int = 23,
                 grow_channels: int = 32,
                 upscale_factor: int = 4):
        """
        The generator of Enhanced SRGAN (ESRGAN).

        Note:
            Network consisting of Residual in Residual Dense Blocks (RRDB).
            It is used as a generator in ESRGAN.
            The number of total parameters: 16,697,987.

        :param in_channels:
            The number of input channels.
        :param out_channels:
            The number of output channels.
        :param n_feats:
            The number of intermediate feature channels.
            Default: 64.
        :param n_basic_blocks:
            The number of basic blocks. (Here basic block is RRDB.)
            Default: 23.
        :param grow_channels:
            The number of channels for each growth in RRDB.
            Default: 32.
        :param upscale_factor:
            Upscale factor.
            Supported values: 2, 3, 4, 8.
        """
        super(RRDBNet, self).__init__()

        # The first conv layer.
        self.first_conv = nn.Conv2d(in_channels, n_feats, kernel_size=3, stride=1, padding=1)

        # The body layers of the network.
        self.body = make_layers(
            ResidualInResidualDenseBlock,
            n_basic_blocks,
            channels=n_feats,  # the argument of RRDB
            grow_channels=grow_channels,  # the argument of RRDB
            res_scale=0.2,  # the argument of RRDB
            esrgan_init=True  # the argument of RRDB
        )
        self.body_conv = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1)

        # The upsample block.
        self.upsample_block = InterpolateUpsampleBlock(n_feats, upscale_factor=upscale_factor)

        # The last 2 conv layers.
        self.hr_conv = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1)
        self.last_conv = nn.Conv2d(n_feats, out_channels, kernel_size=3, stride=1, padding=1)

        # The activation.
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        first_feats = self.first_conv(x)
        body_feats = self.body_conv(self.body(first_feats))
        feats = self.upsample_block(first_feats + body_feats)
        out = self.last_conv(self.lrelu(self.hr_conv(feats)))
        return out


class MSRResNet(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 n_feats: int = 64,
                 n_basic_blocks: int = 16,
                 upscale_factor: int = 4):
        """
        Modified SRResNet.

        Note:
            A compacted version modified from SRResNet in SRGAN.
            It uses Residual Blocks w/o BN, similar to EDSR.
            Currently, it supports x2, x3 and x4 upscale factor.
            The number of total parameters: 1,517,571.

        :param in_channels:
            The number of input channels.
        :param out_channels:
            The number of output channels.
        :param n_feats:
            The number of intermediate feature channels.
            Default: 64.
        :param n_basic_blocks:
            The number of blocks in the body of network.
            Default: 16.
        :param upscale_factor:
            Upscale factor.
            Supported values: 2, 3, 4, 8.
        """
        super(MSRResNet, self).__init__()

        self.upscale_factor = upscale_factor

        # The first conv layer.
        self.first_conv = nn.Conv2d(in_channels, n_feats, kernel_size=3, stride=1, padding=1)

        # The body layers of the network.
        self.body = make_layers(
            ResidualBlockNoBN,
            n_basic_blocks,
            channels=n_feats,  # the argument of RB w/o BN
            esrgan_init=True  # the argument of RB w/o BN
        )

        # The upsample block.
        self.upsample_block = PixelShuffleUpsampleBlock(n_feats, upscale_factor=self.upscale_factor)

        # The last 2 conv layers.
        self.hr_conv = nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1)
        self.last_conv = nn.Conv2d(n_feats, out_channels, kernel_size=3, stride=1, padding=1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights([self.first_conv, self.upsample_block, self.hr_conv, self.last_conv], 0.1)

    def forward(self, x):
        feats = self.lrelu(self.first_conv(x))
        out = self.body(feats)
        out = self.upsample_block(out)
        out = self.last_conv(self.lrelu(self.hr_conv(out)))
        base = nn_f.interpolate(x, scale_factor=self.upscale_factor, mode='bilinear', align_corners=False)
        return base + out


class EnhanceNet(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 3,
                 upscale_factor: int = 4,
                 n_res_blocks: int = 10):
        """
        "EnhanceNet: Single image super-resolution through automated texture synthesis." In ICCV 2017.

        Note:
            The number of total parameters: 889,795.
        """
        super(EnhanceNet, self).__init__()

        assert in_channels == out_channels, "The input and output channels should be equal in EnhanceNet."
        assert upscale_factor in [2, 4, 8], (
            f"The upscale_factor = {upscale_factor} is not supported."
            f"The supported upscale factor values: 2, 4, 8."
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        # Residual blocks.
        res_blocks = [self.__res_blocks(n_feats=64) for _ in range(n_res_blocks)]
        self.res_blocks = nn.Sequential(*res_blocks)

        # Nearest neighbor upsample.
        upsample = []
        for _ in range(int(math.log2(upscale_factor) + 1e-8)):
            upsample.extend([
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True)
            ])
        self.upsample = nn.Sequential(*upsample)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv4 = nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1)

        self.resize = nn.Upsample(scale_factor=4, mode="bicubic", align_corners=True)

    @staticmethod
    def __res_blocks(n_feats):
        return nn.Sequential(
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feats, n_feats, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.res_blocks(out)
        out = self.conv2(out)
        out = self.upsample(out)
        out = self.conv3(out)
        i_res = self.conv4(out)
        i_bicubic = self.resize(x)
        return torch.add(i_bicubic, i_res)


if __name__ == '__main__':
    pass
