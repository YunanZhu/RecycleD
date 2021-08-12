"""
Use some method to optimize the discriminator to get better quality assess ability.

It is constructed on the basis of `archs/weight_map_producer.py`.
"""
from typing import Union

import torch
import torch.nn as nn

from models import srgan
from models.archs import weight_map_producer as wmp, blocks, sod

MAP_TO_MAT_INFO = {
    # The info to help convert weight map to weight matrix.
    # Template: (scale_factor, low_delta, up_delta, left_delta, right_delta).
    # From mat index to map region: (i, j) -> (i * sf + low_d, i * sf + up_d, j * sf + left_d, j * sf + right_d).
    # Note: sf = scale_factor, low_d = low_delta, etc; If the index exceeds the boundary, reset to the boundary value.
    # You can get this info by running "receptive_field.py".
    "PatchVGG": (16, -62, 77, -62, 77),  # Receptive field size: 140.
    "PatchVGG_old": (16, -62, 62, -62, 62),  # Receptive field size: 125. Now this old version has been abandoned.
}

# The pre-trained model file to be loaded by SOD network.
# If you want to use your own SOD network, please add the info here and modify the code in `define_predictor` below.
PRETRAINED_MODEL_FILE = {
    "BASNet": "./pretrained_models/SOD/basnet.pth",  # BASNet.
    "U2Netp": "./pretrained_models/SOD/u2netp.pth",  # Smaller U^2-Net.
    "U2Net": "./pretrained_models/SOD/u2net.pth",  # Full-size U^2-Net.
}


def map_to_mat(weight_map,
               mat_size: Union[list, tuple],
               map_to_mat_info: tuple):
    """
    The converter which can resize weight map to weight matrix.

    :param weight_map:
        The map to be converted.
        Shape: [N, C=1, H, W].
    :param mat_size:
        The height H' and width W' of weight matrix.
    :param map_to_mat_info:
        The info to help convert weight map to weight matrix.
        Template: (scale_factor, low_delta, up_delta, left_delta, right_delta).
        From mat index to map region: (i * sf + low_d, i * sf + up_d, j * sf + left_d, j * sf + right_d).
    :return:
        The resized weight matrix.
        Shape: [N, C=1, H', W'].
    """
    assert len(weight_map.size()) == 4 and weight_map.size(1) == 1, (
        f"Illegal shape of weight map: {weight_map.size()}."
    )

    scale_factor, low_delta, up_delta, left_delta, right_delta = map_to_mat_info

    assert isinstance(scale_factor, int) and scale_factor >= 1, (
        f"The type of scale factor should be {int} rather than {type(scale_factor)}."
    )

    assert isinstance(mat_size, (list, tuple)) and len(mat_size) == 2, f"Illegal mat_size = {mat_size}."

    batch_size, _, map_h, map_w = weight_map.size()
    mat = torch.zeros(
        size=(batch_size, 1, mat_size[0], mat_size[1]),
        dtype=weight_map.dtype,
        device=weight_map.device
    )

    for i in range(mat_size[0]):
        for j in range(mat_size[1]):
            # The center of receptive field.
            rf_center = (i * scale_factor, j * scale_factor)

            # The boundary of receptive field.
            low = max(0, rf_center[0] + low_delta)
            up = min(map_h - 1, rf_center[0] + up_delta)
            left = max(0, rf_center[1] + left_delta)
            right = min(map_w - 1, rf_center[1] + right_delta)

            # Calculate the weight by average operation.
            torch.mean(weight_map[:, :, low:(up + 1), left:(right + 1)], dim=[2, 3], out=mat[:, :, i, j])

    return mat


class Map2Mat_WeightedAveragePredictor(nn.Module):
    def __init__(self,
                 d_model,
                 weight_map_producer,
                 map_mat_converter=None):
        """
        Optimize quality assess ability of PatchGAN discriminator by additional weighted average operation.
        Convert weight map to weight matrix and use weight matrix to provide weights for weighted average operation.

        Deprecate to use this class to wrap the PatchGAN discriminator. (This is the initial version.)
        Recommend to use `SodResFusionTest_WeightedAveragePredictor`.

        :param d_model:
            The pre-trained PatchGAN discriminator model.
            Generally, it is the instance of `nn.Module`.
        :param weight_map_producer:
            Used to produce weight map.
        :param map_mat_converter:
            Used to resize weight map to weight matrix.
            The weight matrix is aligned with quality score matrix output by PatchGAN discriminator.
        """
        super(Map2Mat_WeightedAveragePredictor, self).__init__()
        self.D = d_model
        self.wmp = weight_map_producer
        self.converter = map_mat_converter

    def forward(self, x):
        # Get quality score matrix by PatchGAN discriminator.
        score_mat = self.D(x, patch_result=True, shape="mat4d")

        # Convert weight map to weight matrix.
        if self.converter is not None:
            weight_map = self.wmp(x)
            weight_mat = self.converter(
                weight_map=weight_map,
                mat_size=(score_mat.size(2), score_mat.size(3))  # [mat_H, mat_W]
            )
        else:
            weight_mat = self.wmp(x)

        # Check the shape of weight matrix and quality score matrix.
        assert weight_mat.size() == score_mat.size(), (
            f"The shapes of weight matrix and quality score matrix are not equal."
        )

        return torch.mean(score_mat * weight_mat)


class GivenMat_WeightedAveragePredictor(nn.Module):
    def __init__(self,
                 d_model,
                 weight_mat_producer):
        """
        Directly given the weight MAT producer, use it to boost the performance of PatchGAN discriminator.

        Deprecate to use this class to wrap the PatchGAN discriminator.
        Recommend to use `SodResFusionTest_WeightedAveragePredictor`.

        :param d_model:
            The pre-trained PatchGAN discriminator model.
            Generally, it is the instance of `nn.Module`.
        :param weight_mat_producer:
            Input an image, it can directly output the weight matrix (not weight map).
        """
        super(GivenMat_WeightedAveragePredictor, self).__init__()
        self.D = d_model
        self.wmp = weight_mat_producer

    def forward(self, x):
        # Get quality score matrix by PatchGAN discriminator.
        score_mat = self.D(x, patch_result=True, shape="mat4d")

        # Generate weight matrix by weight_mat_producer.
        weight_mat = self.wmp(x, mat_size=(score_mat.size(2), score_mat.size(3)))

        # Check the shape of weight matrix and quality score matrix.
        assert weight_mat.size() == score_mat.size(), (
            f"The shapes of weight matrix {weight_mat.size()} "
            f"and quality score matrix {score_mat.size()} are not equal."
        )

        return torch.mean(score_mat * weight_mat)


class SodTest_WeightedAveragePredictor(nn.Module):
    def __init__(self,
                 d_model,
                 weight_map_producer,
                 map_mat_converter):
        """
        Given the weight map producer based on the SOD network,
        use it to boost the performance of PatchGAN discriminator.

        Deprecate to use this class to wrap the PatchGAN discriminator.
        Recommend to use `SodResFusionTest_WeightedAveragePredictor`.

        :param d_model:
            The pre-trained PatchGAN discriminator model.
            Generally, it is the instance of `nn.Module`.
        :param weight_map_producer:
            The weight map producer based on the SOD network.
        :param map_mat_converter:
            The function to resize map to matrix.
        """
        super(SodTest_WeightedAveragePredictor, self).__init__()
        self.D = d_model
        self.wmp = weight_map_producer
        self.converter = map_mat_converter

    def forward(self, x):
        # Get quality score matrix by PatchGAN discriminator.
        score_mat = self.D(x, patch_result=True, shape="mat4d")

        # Convert weight map to weight matrix.
        if self.converter is not None:
            weight_map = self.wmp(x)
            weight_mat = self.converter(
                weight_map=weight_map,
                mat_size=(score_mat.size(2), score_mat.size(3))  # [mat_H, mat_W]
            )
        else:
            raise Exception()

        weight_mat = weight_mat + 3.5

        # Check the shape of weight matrix and quality score matrix.
        assert weight_mat.size() == score_mat.size(), (
            f"The shapes of weight matrix {weight_mat.size()} "
            f"and quality score matrix {score_mat.size()} are not equal."
        )

        return torch.mean(score_mat * weight_mat)


class SodResFusionTest_WeightedAveragePredictor(nn.Module):
    def __init__(self,
                 d_model,
                 map_mat_converter,
                 weight_map_producer_sod,
                 weight_map_producer_res,
                 coef_sod: float,
                 coef_res: float,
                 coef_avg: float,
                 norm_type: str = None):
        """
        Use SOD weight mat producer and IR weight mat producer simultaneously to boost the discriminator performance.

        Recommend to use this class to wrap the PatchGAN discriminator.

        :param d_model:
            The pre-trained PatchGAN discriminator model.
            Generally, it is the instance of `nn.Module`.
        :param map_mat_converter:
            The function to resize map to matrix.
        :param weight_map_producer_sod:
            Salient object detection (SOD) weight map producer.
        :param weight_map_producer_res:
            Image Residual (IR) weight map producer.
            Image residual can be regarded as the high-frequency part of an image.
        :param coef_sod:
            The coefficient of SOD weight matrix.
        :param coef_res:
            The coefficient of Image Residual weight matrix.
        :param coef_avg:
            The coefficient of all-ones weight matrix which represents direct average.
            In the paper, it is always set to 1.
        """
        super(SodResFusionTest_WeightedAveragePredictor, self).__init__()

        self.D = d_model
        assert self.D is not None, "The discriminator model cannot be None."

        # The converter will be used to convert weight_map to weight_mat.
        self.converter = map_mat_converter
        assert self.converter is not None, "The converter cannot be None."

        # Check and load the coefficients of 3 weight matrices.
        assert isinstance(coef_sod, float) and isinstance(coef_res, float) and isinstance(coef_avg, float), (
            f"The coefficients should be float numbers,"
            f"but get {type(coef_sod)}, {type(coef_res)} and {type(coef_avg)}."
        )
        assert coef_sod != 0 or coef_res != 0 or coef_avg != 0, (
            f"At least one coefficient is not equal to zero: "
            f"{coef_sod}, {coef_res}, {coef_avg}."
        )
        self.coef_sod = coef_sod
        self.coef_res = coef_res
        self.coef_avg = coef_avg

        # Load the weight map producers.
        self.wmp_sod = weight_map_producer_sod if (self.coef_sod != 0) else None
        self.wmp_res = weight_map_producer_res if (self.coef_res != 0) else None

        assert (norm_type is None) or isinstance(norm_type, str), f"Illegal norm_type type: {type(norm_type)}."
        self.norm_type = norm_type.lower()

    def forward(self, x):
        # Get quality score matrix from the PatchGAN discriminator.
        score_mat = self.D(x, patch_result=True, shape="mat4d")

        # Build an empty weight map.
        if self.coef_sod != 0 or self.coef_res != 0:
            weight_map = torch.zeros(
                size=[x.size(0), 1, x.size(2), x.size(3)],
                dtype=x.dtype, device=x.device,
                requires_grad=False
            )
        else:
            weight_map = None

        # SOD weight map.
        if self.coef_sod != 0:
            weight_map += self.coef_sod * self.wmp_sod(x)

        # Image residual (high-frequency) weight map.
        if self.coef_res != 0:
            weight_map += self.coef_res * self.wmp_res(x)

        # Resize weight map to weight matrix.
        if self.coef_sod != 0 or self.coef_res != 0:
            weight_mat = self.converter(
                weight_map=weight_map,
                mat_size=(score_mat.size(2), score_mat.size(3))  # [mat_H, mat_W]
            ) + self.coef_avg
        else:
            weight_mat = self.coef_avg * torch.ones_like(score_mat)

        # Check the shapes of weight matrix and score matrix.
        assert weight_mat.size() == score_mat.size(), (
            f"The shapes of weight matrix {weight_mat.size()} "
            f"and quality score matrix {score_mat.size()} are not equal."
        )

        # Return the final quality score with shape = [N].
        if (self.norm_type == "mean") or (self.norm_type is None):
            return torch.mean(weight_mat * score_mat, dim=(1, 2, 3), keepdim=False)
        elif self.norm_type == "set_sum_1":
            weight_mat /= torch.sum(weight_mat, dim=(2, 3), keepdim=True)
            return torch.sum(weight_mat * score_mat, dim=(1, 2, 3), keepdim=False)
        else:
            raise NotImplementedError(f"Cannot recognize norm_type: {self.norm_type}.")


def define_predictor(d_choice: str,
                     d_model: nn.Module,
                     in_channels: int,
                     fusion_mode: str,
                     **kwargs):
    import functools
    if fusion_mode == "SR residual weight":  # It is deprecated.
        print(f"Warning: You are using a deprecated fusion_mode = {fusion_mode}.")

        producer = wmp.HighFrequencyResidual(
            in_channels=in_channels,
            scale_factor=4,
            convert_gray=True
        )
        converter = functools.partial(
            map_to_mat,
            map_to_mat_info=MAP_TO_MAT_INFO[d_choice]
        )
        predictor = Map2Mat_WeightedAveragePredictor(
            d_model=d_model,
            weight_map_producer=producer,
            map_mat_converter=converter
        )
    elif fusion_mode == "blur residual weight":  # It has been abandoned.
        raise NotImplementedError(f"The fusion_mode = {fusion_mode} has been abandoned.")

        producer = wmp.HighFrequencyGaussBlur(
            in_channels=3,
            sigma=10,
            kernel_size=7,
            convert_gray=True
        )
        converter = functools.partial(
            map_to_mat,
            map_to_mat_info=MAP_TO_MAT_INFO[d_choice]
        )
        predictor = Map2Mat_WeightedAveragePredictor(
            d_model=d_model,
            weight_map_producer=producer,
            map_mat_converter=converter
        )
    elif fusion_mode == "DFT high-freq weight":  # It has been abandoned.
        raise NotImplementedError(f"The fusion_mode = {fusion_mode} has been abandoned.")

        producer = wmp.HighFrequencyDFT(
            in_channels=3,
            map_to_mat_info=MAP_TO_MAT_INFO[d_choice]
        )
        predictor = GivenMat_WeightedAveragePredictor(
            d_model=d_model,
            weight_mat_producer=producer
        )
    elif fusion_mode == "SOD BASNet weight":  # It is deprecated.
        print(f"Warning: You are using a deprecated fusion_mode = {fusion_mode}.")

        producer = wmp.SodBASNet(PRETRAINED_MODEL_FILE["BASNet"])
        converter = functools.partial(
            map_to_mat,
            map_to_mat_info=MAP_TO_MAT_INFO[d_choice]
        )
        predictor = SodTest_WeightedAveragePredictor(
            d_model=d_model,
            weight_map_producer=producer,
            map_mat_converter=converter
        )
    elif fusion_mode == "SOD U2Net weight":  # It is deprecated.
        print(f"Warning: You are using a deprecated fusion_mode = {fusion_mode}.")

        producer = wmp.SodU2Net(
            full_size=True,
            model_file=PRETRAINED_MODEL_FILE["U2Net"]
        )
        converter = functools.partial(
            map_to_mat,
            map_to_mat_info=MAP_TO_MAT_INFO[d_choice]
        )
        predictor = SodTest_WeightedAveragePredictor(
            d_model=d_model,
            weight_map_producer=producer,
            map_mat_converter=converter
        )
    elif fusion_mode == "SOD U2Netp weight":  # It is deprecated.
        print(f"Warning: You are using a deprecated fusion_mode = {fusion_mode}.")

        producer = wmp.SodU2Net(
            full_size=False,
            model_file=PRETRAINED_MODEL_FILE["U2Netp"]
        )
        converter = functools.partial(
            map_to_mat,
            map_to_mat_info=MAP_TO_MAT_INFO[d_choice]
        )
        predictor = SodTest_WeightedAveragePredictor(
            d_model=d_model,
            weight_map_producer=producer,
            map_mat_converter=converter
        )
    elif fusion_mode == "SOD U2Net + SR residual" or fusion_mode == "SOD BASNet + SR residual":  # It is recommended.
        # Load SOD weight map producer.
        if "SOD U2Net" in fusion_mode:
            producer1 = wmp.SodU2Net(
                full_size=True,
                model_file=PRETRAINED_MODEL_FILE["U2Net"]
            )
        elif "SOD BASNet" in fusion_mode:
            producer1 = wmp.SodBASNet(PRETRAINED_MODEL_FILE["BASNet"])
        else:
            raise NotImplementedError(f"Cannot recognize the SOD model type in {fusion_mode}.")

        # Load IR weight map producer.
        producer2 = wmp.HighFrequencyResidual(
            in_channels=in_channels,
            scale_factor=4,  # You can modify it if you want to test other scale_factor.
            convert_gray=True
        )

        # It is used to convert weight map to weight mat.
        converter = functools.partial(
            map_to_mat,
            map_to_mat_info=MAP_TO_MAT_INFO[d_choice]
        )

        # The coefficients of 3 weight matrices.
        coef_sod = kwargs["coef_sod"] if ("coef_sod" in kwargs.keys()) else None  # The SOD weight matrix.
        coef_res = kwargs["coef_res"] if ("coef_res" in kwargs.keys()) else None  # The Image Residual weight matrix.
        coef_avg = kwargs["coef_avg"] if ("coef_avg" in kwargs.keys()) else None  # The matrix of direct average.
        print(f"The coefficients of weight matrices: "
              f"coef_sod = {coef_sod}, coef_res = {coef_res}, coef_avg = {coef_avg}.")

        # The post-processing type of total weight mat.
        norm_tpye = kwargs["norm_type"] if ("norm_type" in kwargs.keys()) else None
        print(f"The normalization type of the total weight matrix: {norm_tpye}.")

        # Build the final predictor.
        predictor = SodResFusionTest_WeightedAveragePredictor(
            d_model=d_model,
            weight_map_producer_sod=producer1,
            weight_map_producer_res=producer2,
            map_mat_converter=converter,
            coef_sod=coef_sod,
            coef_res=coef_res,
            coef_avg=coef_avg,
            norm_type=norm_tpye
        )
    else:
        raise ValueError(f"Cannot recognize fusion_mode = {fusion_mode}.")

    return predictor


if __name__ == '__main__':
    pass
