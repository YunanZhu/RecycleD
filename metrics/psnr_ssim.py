"""
Calculate PSNR and SSIM of tensors.
"""
import math
import cv2

import torch
import torch.nn.functional as nn_f

# The weights of R/G/B channels. Convert from RGB to Y channel.
# Ref: https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.
RGB_W_FOR_Y = torch.as_tensor([0.299, 0.587, 0.114]).view(3, 1, 1)


def tensor_psnr(image1, image2, test_y_channel=True):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio).

    It returns the same result as `skimage.metrics.peak_signal_noise_ratio`.

    :param image1:
        A tensor with shape [C, H, W] and data range [0,1].
    :param image2:
        Another tensor with shape [C, H, W] and data range [0,1].
    :param test_y_channel:
        If test on Y channel of YCbCr.
        It is effective only when inputs are RGB images.
    """
    assert len(image1.size()) == 3 and len(image2.size()) == 3, (
        f"Illegal image tensor dims: {len(image1.size())}, {len(image2.size())}."
    )
    assert image1.size() == image2.size(), f"Different image shapes: {image1.size()}, {image2.size()}."
    assert image1.size(0) == 1 or image1.size(0) == 3, (
        f"Illegal number of channels: {image1.size(0)}. It must be a RGB or grayscale image."
    )

    if test_y_channel and image1.size(0) == 3:
        image1 = torch.sum(image1 * RGB_W_FOR_Y.to(image1), dim=0, keepdim=False)
        image2 = torch.sum(image2 * RGB_W_FOR_Y.to(image2), dim=0, keepdim=False)

    mse = nn_f.mse_loss(image1, image2, reduction="mean").item()
    if mse > 0:
        return 10.0 * math.log10(1.0 / mse)
    else:
        return float("inf")


def tensor_ssim(image1, image2, test_y_channel=False):
    """
    Calculate SSIM (structural similarity).

    Need to use function `skimage.metrics.structural_similarity`.
    It need to move the data from GPU to CPU, so it takes a lot of time.

    :param image1:
        A tensor with shape [C, H, W] and data range [0, 1].
    :param image2:
        A tensor with shape [C, H, W] and data range [0, 1].
    :param test_y_channel:
        If test on Y channel of YCbCr.
        It is effective only when inputs are RGB images.
    """
    from skimage.metrics import structural_similarity as ssim

    assert len(image1.size()) == 3 and len(image2.size()) == 3, (
        f"Illegal image dims: {len(image1.size())}, {len(image2.size())}."
    )
    assert image1.size() == image2.size(), f"Different image shapes: {image1.size()}, {image2.size()}."
    assert image1.size(0) == 1 or image1.size(0) == 3, (
        f"Illegal number of channels: {image1.size(0)}. It must be a RGB or grayscale image."
    )

    # If inputs are RGB images.
    if image1.size(0) == 3 and image2.size(0) == 3:
        if test_y_channel:
            image1 = torch.sum(image1 * RGB_W_FOR_Y.to(image1), dim=0, keepdim=False)
            image2 = torch.sum(image2 * RGB_W_FOR_Y.to(image2), dim=0, keepdim=False)
        else:
            image1, image2 = image1.permute(1, 2, 0), image2.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]
    # If inputs are grayscale images.
    else:
        image1 = image1.squeeze()
        image2 = image2.squeeze()

    # From GPU to CPU.
    image1 = image1.detach().cpu()
    image2 = image2.detach().cpu()

    if len(image1.size()) == 2 and len(image2.size()) == 2:
        return ssim(image1.numpy(), image2.numpy(), data_range=1, multichannel=False)
    else:
        return ssim(image1.numpy(), image2.numpy(), data_range=1, multichannel=True)


if __name__ == '__main__':
    pass
