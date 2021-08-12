import torch
import torch.nn as nn

from models.archs.arch_utils import *


# ------------------ VGG-Style Discriminator (current) ------------------
def discriminator_block(in_channels: int, out_channels: int) -> list:
    """
    Generate a basic block of VGG-style discriminator.

    Note:
        Don't use BN layer in discriminator if use WGAN-GP.
        H => floor(H / 2), W => floor(W / 2).

    :param in_channels:
        The number of input channels.
    :param out_channels:
        The number of output channels.
    :return:
        A list of conv/activation layers.
    """
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(negative_slope=0.2, inplace=True)
    ]


class VggDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3, use_msra_init: bool = True):
        """
        VGG-style discriminator.
        It is used to train SRGAN/ESRGAN.

        Note:
            Don't use BN layer in discriminator if use WGAN-GP.
            The number of total parameters: 7,941,897.
            Recommended input size: 192 * 192.
        
        :param in_channels:
            The number of input channels.
        :param use_msra_init:
            If use MSRA initialization that is also mentioned in ESRGAN.
        """
        super(VggDiscriminator, self).__init__()

        layers = []
        layers.extend(discriminator_block(in_channels, 64))  # 192 * 192 => 96 * 96
        layers.extend(discriminator_block(64, 128))  # => 48 * 48
        layers.extend(discriminator_block(128, 256))  # => 24 * 24
        layers.extend(discriminator_block(256, 512))  # => 12 * 12

        layers.extend([
            nn.AdaptiveAvgPool2d(output_size=(4, 4)),  # 12 * 12 => 4 * 4
            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 100),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(100, 1)
        ])

        if use_msra_init:
            default_init_weights(layers)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class PatchVggDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3, use_msra_init: bool = True):
        """
        VGG-style PatchGAN discriminator.
        It is used to train SRGAN/ESRGAN.

        Note:
            Don't use BN layer in discriminator if use WGAN-GP.
            The number of total parameters: 8,009,153.
            Receptive field size of one output pixel: 140 * 140.
            Recommended input size: 192 * 192.

        :param in_channels:
            The number of input channels.
        :param use_msra_init:
            If use MSRA initialization that is also mentioned in ESRGAN.
        """
        super(PatchVggDiscriminator, self).__init__()

        layers = []
        layers.extend(discriminator_block(in_channels, 64))  # 192 * 192 => 96 * 96
        layers.extend(discriminator_block(64, 128))  # => 48 * 48
        layers.extend(discriminator_block(128, 256))  # => 24 * 24
        layers.extend(discriminator_block(256, 512))  # => 12 * 12

        layers.extend([
            nn.Conv2d(512, 192, kernel_size=3, stride=1, padding=1, bias=True),  # => 12 * 12
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(192, 1, kernel_size=3, stride=1, padding=1, bias=True)  # => 12 * 12
        ])

        if use_msra_init:
            default_init_weights(layers)

        self.model = nn.Sequential(*layers)

    def forward(self, x, patch_result: bool = False, shape: str = "mat4d"):
        """
        Forward pass.

        :param x:
            The network input.
        :param patch_result:
            Whether return a real number or not.
            When input is one image (batch_size = 1),
            Return a real number of the input image if set to False.
            Otherwise return a matrix/vector (In it, each element is associated with a patch on the input image.).
        :param shape:
            It allows you to control the output shape.
        """
        if patch_result:
            return self.get_patch_result(x, shape)
        else:
            out = self.model(x)
            return out.mean(dim=(2, 3), keepdim=False)

    def get_patch_result(self, x, shape: str = "mat4d"):
        """
        Used to serve the function `forward`.
        It allows you to control the output shape.

        :param x:
            The network input.
        :param shape:
            A string to control the output shape.
            Supported values: "mat4d", "mat3d", "vec3d" and "vec2d".
        """
        out = self.model(x)

        if shape == "mat4d":  # [N, C=1, H, W]
            return out
        elif shape == "mat3d":  # [N, H, W]
            return out.squeeze(1)
        elif shape == "vec3d":  # [N, C=1, H * W]
            return out.view(out.size(0), out.size(1), -1)
        elif shape == "vec2d":  # [N, H * W]
            return out.view(out.size(0), -1)
        else:
            raise NotImplementedError(f"The input shape control = {shape} can't be recognized.")


# ------------------ Deeper VGG-Style Discriminator (17 weight layers) ------------------
class DeeperVggDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3, use_msra_init: bool = True):
        """
        Deeper VGG-style Discriminator.
        It is used to compare with the ordinary VGG-style Discriminator.

        Note:
            It has more parameters than ordinary VGG-style Discriminator.
            The number of total parameters: 12,143,561.
            Recommended input size: 192 * 192.

        :param in_channels:
            The number of input channels.
        :param use_msra_init:
            If use MSRA initialization that is also mentioned in ESRGAN.
        """
        super(DeeperVggDiscriminator, self).__init__()

        layers = []
        layers.extend(self.__block(in_channels, 32))  # 192 * 192 => 96 * 96
        layers.extend(self.__block(32, 64))  # => 48 * 48
        layers.extend(self.__block(64, 128))  # => 24 * 24
        layers.extend(self.__block(128, 256))  # => 12 * 12
        layers.extend(self.__block(256, 512))  # => 6 * 6

        layers.extend([
            nn.AdaptiveAvgPool2d(output_size=(6, 6)),  # 6 * 6 => 6 * 6
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 100),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(100, 1)
        ])

        if use_msra_init:
            default_init_weights(layers)

        self.model = nn.Sequential(*layers)

    @staticmethod
    def __block(in_channels: int, out_channels: int) -> list:
        """
        It is similar to `discriminator_block` but with one more conv layer.
        """
        return [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        ]

    def forward(self, x):
        return self.model(x)


# ------------------ ResNet-18 Discriminator ------------------
class BasicResBlock(nn.Module):
    """It is copied from `torchvision.models`."""
    expansion = 1

    @staticmethod
    def __conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
        """3x3 convolution with padding"""
        return nn.Conv2d(
            in_planes, out_planes,
            kernel_size=3, stride=stride, padding=dilation,
            dilation=dilation, groups=groups, bias=False
        )

    def __init__(self, inplanes, planes,
                 stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        """It is only used for ResNet Discriminator."""
        super(BasicResBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = self.__conv3x3(inplanes, planes, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.__conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetDiscriminator(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 1):
        """
        ResNet-style discriminator (ResNet-18).
        It is used to compare with the ordinary VGG-style Discriminator.

        Note:
            It has a different architecture with the ordinary VGG-style Discriminator.
            The number of total parameters: 11,167,425.
            Recommended input size: 192 * 192.
        """
        super(ResNetDiscriminator, self).__init__()
        block = BasicResBlock

        self.inplanes = 64
        self.dilation = 1

        self.groups = 1
        self.base_width = 64
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, 2)
        self.layer2 = self._make_layer(block, 128, 2, stride=2)
        self.layer3 = self._make_layer(block, 256, 2, stride=2)
        self.layer4 = self._make_layer(block, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, n_blocks, stride=1):
        downsample = None
        previous_dilation = self.dilation
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Conv2d(
                self.inplanes, planes * block.expansion,
                kernel_size=1, stride=stride,
                bias=False
            )

        layers = [block(
            self.inplanes, planes, stride, downsample,
            self.groups, self.base_width,
            previous_dilation
        )]
        self.inplanes = planes * block.expansion
        for _ in range(1, n_blocks):
            layers.append(
                block(
                    self.inplanes, planes, groups=self.groups,
                    base_width=self.base_width, dilation=self.dilation
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    pass
