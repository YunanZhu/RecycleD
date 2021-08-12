"""
Used to define the generator and discriminator.
It is based on `archs/generator.py` and `archs/discriminator.py`.
"""


def define_generator(choice: str,
                     in_channels: int,
                     out_channels: int,
                     upscale_factor: int,
                     **kwargs):
    """
    Return a generator model.

    :param choice:
        The choice of generator.
        Support: "RRDBNet", "MSRResNet".
    :param in_channels:
        The number of input channels.
    :param out_channels:
        The number of output channels.
    :param upscale_factor:
        Upscale factor.
        Support: 2, 3, 4, 8.
    """
    from models.archs import generator, arch_utils

    if choice == "RRDBNet":
        net = generator.RRDBNet(
            in_channels, out_channels,
            upscale_factor=upscale_factor
        )
    elif choice == "MSRResNet":
        net = generator.MSRResNet(
            in_channels, out_channels,
            upscale_factor=upscale_factor
        )
    elif choice == "SRResNet_old":
        net = generator.SRResNet_OldVersion(
            in_channels, out_channels,
            upscale_factor=upscale_factor
        )
    elif choice == "EnhanceNet":
        net = generator.EnhanceNet(
            in_channels, out_channels,
            upscale_factor=upscale_factor
        )
    else:
        raise NotImplementedError(f"The generator choice = {choice} is illegal.")

    if "count_params" in kwargs.keys() and kwargs["count_params"]:
        print(f"The parameters of {choice}:")
        arch_utils.count_params(net)

    return net


def define_discriminator(choice: str,
                         in_channels: int,
                         **kwargs):
    """
    Return a discriminator model.

    :param choice:
        The choice of discriminator.
        Support: "VGG", "PatchVGG".
    :param in_channels:
        The number of input channels.
    """
    from models.archs import discriminator, arch_utils

    if choice == "VGG":
        net = discriminator.VggDiscriminator(
            in_channels,
            use_msra_init=True
        )
    elif choice == "PatchVGG":
        net = discriminator.PatchVggDiscriminator(
            in_channels,
            use_msra_init=True
        )
    elif choice == "VGG_old":
        raise NotImplementedError(f"This old version {choice} has been abandoned.")
    elif choice == "PatchVGG_old":
        raise NotImplementedError(f"This old version {choice} has been abandoned.")
    elif choice == "DeeperVGG":
        net = discriminator.DeeperVggDiscriminator(
            in_channels,
            use_msra_init=True
        )
    elif choice == "ResNet18":
        net = discriminator.ResNetDiscriminator(
            in_channels,
            num_classes=1
        )
    else:
        raise NotImplementedError(f"The discriminator choice = {choice} is illegal.")

    if "count_params" in kwargs.keys() and kwargs["count_params"]:
        print(f"The parameters of {choice}:")
        arch_utils.count_params(net)

    return net
