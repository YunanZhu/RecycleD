"""
An naive method to calculate the receptive field of discriminator.
The receptive field of discriminator is useful in `MAP_TO_MAT_INFO` in `models/optim_predictor.py`.
"""
import torch
import torch.nn as nn

from models import srgan


def main():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("cuda state: " + "available" if use_cuda else "unavailable")  # Check if cuda is available.

    D = srgan.define_discriminator(
        "PatchVGG",
        in_channels=1,
        count_params=True
    ).to(device)

    x = torch.randn(size=[1, 1, 321, 480]).to(device).requires_grad_(True)
    y = D.get_patch_result(x)

    print(f"The shape of input: {x.size()}, shape of output: {y.size()}.")

    grad_outputs = torch.zeros_like(y)
    grad_outputs[0, 0, 0, 29] = 1.0

    gradients = torch.autograd.grad(
        outputs=y, inputs=x,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    min_i, min_j = int(1e8), int(1e8)
    max_i, max_j = -1, -1
    for i in range(0, x.size(2)):
        for j in range(0, x.size(3)):
            if gradients[0, 0, i, j] != 0:
                min_i, min_j = min(min_i, i), min(min_j, j)
                max_i, max_j = max(max_i, i), max(max_j, j)

    print(f"i: [bottom, top] = [{min_i}, {max_i}], len = {max_i - min_i + 1}, mid = {(min_i + max_i) / 2}")
    print(f"j: [left, right] = [{min_j}, {max_j}], len = {max_j - min_j + 1}, mid = {(min_j + max_j) / 2}")


if __name__ == '__main__':
    main()
