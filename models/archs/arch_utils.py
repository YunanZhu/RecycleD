import math

import torch
from torch import nn
from torch.nn import functional as nn_f
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm


@torch.no_grad()
def default_init_weights(module_list,
                         scale: float = 1,
                         bias_fill: float = 0,
                         **kwargs) -> None:
    """
    Initialize network weights.

    I copy this function from xinntao's BasicSR: https://github.com/xinntao/BasicSR.

    :param module_list:
        Modules to be initialized.
        Type: `list[nn.Module]` or `nn.Module`.
    :param scale:
        Scale initialized weights, especially for residual blocks.
    :param bias_fill:
        The value to fill bias.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]

    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layers(basic_block,
                n_basic_blocks: int,
                **kwarg) -> nn.Sequential:
    """
    Make layers by stacking the same basic blocks.

    :param basic_block:
        Basic block to be stacked.
    :param n_basic_blocks:
        The number of blocks.
    :param kwarg:
        The arguments for constructing basic blocks.
    :return:
        Stacked basic blocks in `nn.Sequential`.
    """
    assert isinstance(n_basic_blocks, int), f"The n_basic_blocks type should be {int}."
    layers = [basic_block(**kwarg) for _ in range(n_basic_blocks)]
    return nn.Sequential(*layers)


def count_params(model: nn.Module, verbose: bool = True):
    """
    A simple function to calculate the number of total parameters and trainable parameters of a model.

    :param model:
        The model to be calculated the number of parameters.
    :param verbose:
        Print the result if set to true.
    """
    assert isinstance(model, nn.Module), f"The type of input model should be {nn.Module} rather than {type(model)}."

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if verbose:
        print(f"{model.__class__.__name__}: "
              f"{total_params:,} total parameters; "
              f"{total_trainable_params:,} trainable parameters.")

    return total_params, total_trainable_params


if __name__ == '__main__':
    pass
