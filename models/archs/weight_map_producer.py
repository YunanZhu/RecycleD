"""
The architecture of weight map producer.

The weight map producers can be classified into 2 categories:
    1. Based on salient object detection.
    2. Based on image residuals.
"""
import math
from typing import Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as nn_f


class HighFrequencyResidual(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 scale_factor: int = 2,
                 mode: str = "bicubic",
                 convert_gray: bool = True):
        """
        A weight map producer based on the image residuals.

        Ref:
            Yi et al. Contextual Residual Aggregation for Ultra High-Resolution Image Inpainting. In CVPR 2020.

        Note:
            Get high frequency residual by subtract HR and SR image.
            LR := downsample(HR), SR := upsample(LR).

        :param in_channels:
            The number of input channels.
        :param scale_factor:
            The scale ratio between HR and LR.
        :param mode:
            Downsampling and upsampling algorithm.
            Ref `torch.nn.functional.interpolate()`.
        :param convert_gray:
            If convert input image to grayscale at first.
        """
        super(HighFrequencyResidual, self).__init__()
        self.in_channels = in_channels
        self.scale_factor = scale_factor
        self.mode = mode

        self.convert_gray = convert_gray
        if self.convert_gray:
            assert self.in_channels == 3, f"Can only convert RGB to gray, but in_channels = {self.in_channels}."

            # Ref ITU-R BT.601 conversion: https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.
            # RGB to YCbCr: Y = 0.299 * R + 0.587 * G + 0.114 * B.
            rgb_weight = torch.as_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            self.register_buffer("rgb_weight", rgb_weight)

    def forward(self, x):
        assert x.size(1) == self.in_channels, (
            f"The channels number of input should be {self.in_channels} rather than {x.size(1)}."
        )

        if self.convert_gray:
            x = torch.sum(x * self.rgb_weight, dim=1, keepdim=True)

        lr_x = nn_f.interpolate(x, scale_factor=1 / self.scale_factor, mode=self.mode, align_corners=False)
        sr_x = nn_f.interpolate(lr_x, size=(x.size(2), x.size(3)), mode=self.mode, align_corners=False)

        return torch.abs(x - sr_x)


class HighFrequencyGaussBlur(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 sigma: float = 1.0,
                 kernel_size: int = None,
                 convert_gray: bool = True):
        """
        Get high frequency residual by subtract HR and Gaussian blurred image.

        NOTE: Useless now.

        :param in_channels:
            The number of input channels.
        :param sigma:
            The standard deviation of Gaussian distribution.
        :param kernel_size:
            The size of Gaussian blur kernel.
        :param convert_gray:
            If convert input image to grayscale at first.
        """
        super(HighFrequencyGaussBlur, self).__init__()
        self.in_channels = in_channels

        self.convert_gray = convert_gray
        if self.convert_gray:
            assert self.in_channels == 3, f"Can only convert RGB to gray, but in_channels = {self.in_channels}."
            # Ref ITU-R BT.601 conversion: https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion
            # RGB to YCbCr: Y = 0.299 * R + 0.587 * G + 0.114 * B
            rgb_weight = torch.as_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            self.register_buffer("rgb_weight", rgb_weight)

        kernel = torch.from_numpy(
            self.gauss_kernel(sigma=sigma, size=kernel_size)
        ).to(dtype=torch.float).unsqueeze(0).unsqueeze(0)
        if self.in_channels > 1 and not self.convert_gray:
            kernel = kernel.expand(self.in_channels, -1, -1, -1)  # set out_C := in_C
        self.register_buffer("kernel", kernel)

    def forward(self, x):
        assert x.size(1) == self.in_channels, (
            f"The channels number of input should be {self.in_channels} rather than {x.size(1)}."
        )

        if self.convert_gray:
            x = torch.sum(x * self.rgb_weight, dim=1, keepdim=True)

        if self.kernel.size(-1) % 2 == 1:
            pad_num = (self.kernel.size(-1) - 1) // 2
        else:
            pad_num = self.kernel.size(-1) // 2

        blur_x = nn_f.conv2d(
            x, self.kernel,
            stride=1, padding=pad_num,
            groups=1 if self.convert_gray else self.in_channels
        )[0:x.size(0), 0:x.size(1), 0:x.size(2), 0:x.size(3)]

        return torch.abs(x - blur_x)

    @staticmethod
    def gauss_kernel(sigma: float = 1.0, size: int = None):
        """
        Generate a 2D Gaussian blur kernel.

        :param sigma:
            Standard deviation of Gaussian distribution.
        :param size:
            The size/bandwidth of kernel.
        """
        # Check whether dimension is legal or not.
        import cv2
        kx = cv2.getGaussianKernel(size, sigma)
        ky = cv2.getGaussianKernel(size, sigma)
        return np.multiply(kx, np.transpose(ky))


class HighFrequencyDFT(nn.Module):
    def __init__(self,
                 in_channels,
                 map_to_mat_info):
        """
        Get image residual by subtract HR and DFT low-pass image.

        Note: Useless now.
        """
        super(HighFrequencyDFT, self).__init__()
        self.in_channels = in_channels

        if self.in_channels == 1:  # grayscale image
            self.convert_gray = False
        elif self.in_channels == 3:  # RGB image
            self.convert_gray = True
            # Ref ITU-R BT.601 conversion: https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion
            # RGB to YCbCr: Y = 0.299 * R + 0.587 * G + 0.114 * B
            rgb_weight = torch.as_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
            self.register_buffer("rgb_weight", rgb_weight)
        else:
            raise NotImplementedError(
                f"Can only handle grayscale image or RGB image, but in_channels = {self.in_channels}."
            )

        self.map_to_mat_info = map_to_mat_info

    def forward(self, x, mat_size):
        N, C, H, W = x.size()
        assert N == 1, (
            f"Can only handle batch with size = 1, but input batch size = {N}."
        )
        assert C == self.in_channels, (
            f"The channels number of input should be {self.in_channels} rather than {C}."
        )

        mat = torch.zeros(
            size=(N, 1, mat_size[0], mat_size[1]),
            dtype=x.dtype,
            device=x.device
        )

        if self.convert_gray:
            x = torch.sum(x * self.rgb_weight, dim=1, keepdim=True)

        scale_factor, low_delta, up_delta, left_delta, right_delta = self.map_to_mat_info

        for i in range(mat_size[0]):
            for j in range(mat_size[1]):
                # The center of receptive field.
                rf_center = (i * scale_factor, j * scale_factor)

                # The bound of receptive field.
                low = max(0, rf_center[0] + low_delta)
                up = min(H - 1, rf_center[0] + up_delta)
                left = max(0, rf_center[1] + left_delta)
                right = min(W - 1, rf_center[1] + right_delta)

                mat[0, 0, i, j] = self._one_patch(x[:, :, low:(up + 1), left:(right + 1)])

        return mat / mat.max()

    @staticmethod
    def _one_patch(patch):
        complex_f = torch.rfft(patch, signal_ndim=2, onesided=False)  # [N, C, H, W] -> [N, C, H, W, 2]

        power = torch.pow(complex_f[:, :, :, :, 0], 2) + torch.pow(complex_f[:, :, :, :, 1], 2)
        log_power = torch.log(power + 1)

        batch_size, _, h, w = log_power.size()
        assert h >= 4 and w >= 4, f"The input image (H = {h}, W = {w}) is too small."
        hf_region_low, hf_region_up = h // 2 - h // 4 - 1, h // 2 + h // 4
        hf_region_left, hf_region_right = w // 2 - w // 4 - 1, w // 2 + w // 4

        region1_sum = torch.sum(
            log_power[:, :, hf_region_low:(hf_region_up + 1), :],
            dim=[2, 3], keepdim=False
        )
        region2_sum = torch.sum(
            log_power[:, :, (hf_region_up + 1):, hf_region_left:(hf_region_right + 1)],
            dim=[2, 3], keepdim=False
        )
        region3_sum = torch.sum(
            log_power[:, :, :hf_region_low, hf_region_left:(hf_region_right + 1)],
            dim=[2, 3], keepdim=False
        )
        hf_region_sum = region1_sum + region2_sum + region3_sum

        return hf_region_sum.item()


class SodBASNet(nn.Module):
    def __init__(self, model_file: str = ""):
        """
        Build a weight map producer based on the salient object detector BASNet.
        """
        super(SodBASNet, self).__init__()
        from models.archs import sod

        self.sod_model = sod.BASNet(3, 1)
        self.sod_model.load_state_dict(torch.load(model_file))
        self.sod_model.eval()

        # Turn off requires_grad
        for p in self.sod_model.parameters():
            p.requires_grad_(False)

        mean = torch.as_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.as_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        N, C, H, W = x.size()
        assert C == 3, (
            f"The channels number of input should be 3 rather than {x.size(1)}."
        )
        x = nn_f.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        sod_map = self.sod_model(x)[0]  # The first element is SOD map.
        sod_map = nn_f.interpolate(sod_map, size=(H, W), mode="bilinear", align_corners=False)  # back to original size
        return self.__min_max_norm(sod_map)

    @staticmethod
    def __min_max_norm(x):
        maxi = torch.max(x)
        mini = torch.min(x)
        return (x - mini) / (maxi - mini)


class SodU2Net(nn.Module):
    def __init__(self, full_size: bool = True, model_file: str = ""):
        """
        Build a weight map producer based on the salient object detector U^2-Net.

        :param full_size:
            If use the full-size U^2-Net.
        :param model_file:
            The pre-trained model file.
        """
        super(SodU2Net, self).__init__()
        from models.archs import sod

        if full_size:
            self.sod_model = sod.U2NET(3, 1)
        else:
            self.sod_model = sod.U2NETP(3, 1)

        self.sod_model.load_state_dict(torch.load(model_file))
        self.sod_model.eval()

        # Turn off requires_grad
        for p in self.sod_model.parameters():
            p.requires_grad_(False)

        mean = torch.as_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.as_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, x):
        N, C, H, W = x.size()
        assert C == 3, (
            f"The channels number of input should be 3 rather than {x.size(1)}."
        )
        x = nn_f.interpolate(x, size=(320, 320), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        sod_map = self.sod_model(x)[0]  # The first element is SOD map.
        sod_map = nn_f.interpolate(sod_map, size=(H, W), mode="bilinear", align_corners=False)  # back to original size
        return self.__min_max_norm(sod_map)

    @staticmethod
    def __min_max_norm(x):
        maxi = torch.max(x)
        mini = torch.min(x)
        return (x - mini) / (maxi - mini)


def plot_map_from_image(filename: str, model: nn.Module, device: Union[str, torch.device]):
    """
    Use the pre-trained weight map producer to get weight map from an image, and then display this weight map.
    You can use it to check the performance of the weight map producer.

    :param filename:
        The filename of the image.
    :param model:
        The pre-trained WMP model.
    :param device:
        Used to control the device of the image tensor.
    """
    from PIL import Image
    import torchvision.transforms.functional as transforms_f
    image = Image.open(filename).convert("RGB")
    image = transforms_f.to_tensor(image).to(device=device).unsqueeze_(0)
    w_map = model(image).squeeze_()
    print(f"The sum of W mat: {w_map.sum().item()}.")
    w_map = w_map.mul_(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8).numpy()
    Image.fromarray(w_map).convert('L').show()


if __name__ == '__main__':
    # device = "cuda"
    #
    # u2_net = SodU2Net(full_size=True, model_file=r"E:\Research\WSRGAN\pretrained_models\SOD/u2net.pth").to(device)
    # plot_map_from_image(r"E:\Resources\datasets\PIPAL\train/Distortion_3\A0150_02_01.bmp", u2_net, device)
    #
    # img_res_model = HighFrequencyResidual(in_channels=3, scale_factor=4, convert_gray=True).to(device)
    # plot_map_from_image(r"E:\Resources\datasets\KonIQ-10K\1024x768/826373.jpg", img_res_model, device)
    pass
