import torch
import torch.nn as nn

from torchvision.models.vgg import vgg19


class PerceptualLoss(nn.Module):
    def __init__(self,
                 pretrained_model: str,
                 last_layer: int = 35,
                 criterion_type: str = "L1",
                 loss_weight: float = 1.0):
        """
        Perceptual loss with commonly used VGG feature extractor.

        :param pretrained_model:
            The pretrained model file of VGG19.
        :param last_layer:
            The index of last layer to extract features.
        :param loss_weight:
            The weight of perceptual loss.
            Default: 1.0.
        :param criterion_type:
            Criterion used for perceptual loss.
            Default: "L1".
        """
        super(PerceptualLoss, self).__init__()

        self.feature_extractor = VggFeatureExtractor(
            pretrained_model,
            last_layer=last_layer,
            use_input_norm=True,
            requires_grad=False
        )

        if criterion_type == "L1":
            self.criterion = nn.L1Loss(reduction="mean")
        elif criterion_type == "L2":
            self.criterion = nn.MSELoss(reduction="mean")
        else:
            raise NotImplementedError(
                f"The criterion {criterion_type} is illegal."
            )

        self.loss_weight = loss_weight

    def forward(self, x, gt):
        """
        Forward pass.

        :param x:
            A tensor, input tensor with shape [N, C, H, W].
        :param gt:
            A tensor, ground-truth tensor with shape [N, C, H, W].
        :return:
            A tensor, forward results.
        """
        # Extract VGG features.
        x_features = self.feature_extractor(x)
        gt_features = self.feature_extractor(gt.detach())

        # Calculate perceptual loss.
        if self.loss_weight > 0:
            perceptual_loss = self.criterion(x_features, gt_features) * self.loss_weight
        else:
            perceptual_loss = torch.zeros(1).to(x)

        return perceptual_loss


class VggFeatureExtractor(nn.Module):
    def __init__(self,
                 pretrained_model: str,
                 last_layer: int = 35,
                 use_input_norm: bool = True,
                 requires_grad: bool = False):
        """
        VGG19 network for feature extraction.
        Here it is only used to build the perceptual loss.

        :param pretrained_model:
            The model file to load.
            If set to None, it will use the model file from torchvision.
        :param last_layer:
            The index of last layer to extract features.
            Some possible options:
                VGG19/(5,4): [0:36] (after activation), [0:35] (before activation).
                VGG19/pool4: [0:28].
                VGG19/(3,4): [0:18] (after activation), [0:17] (before activation).
                VGG19/(2,2): [0:9] (after activation), [0:8] (before activation).
            Default: 35.
            Follow the ESRGAN setting use VGG19/(5,4) before activation.
        :param use_input_norm:
            If set to True, normalize the input image according to the tips of Pytorch.
            The input image must be in the value range [0,1] for this normalization.
            Default: True.
        :param requires_grad:
            If set to True, the parameters of VGG network can be optimized.
            Default: False.
        """
        super(VggFeatureExtractor, self).__init__()

        if pretrained_model is None:
            vgg19_model = vgg19(pretrained=True, progress=True)
        else:
            vgg19_model = vgg19(pretrained=False, progress=False)
            vgg19_model.load_state_dict(torch.load(pretrained_model))

        self.feature_extractor = nn.Sequential(
            *list(vgg19_model.features)[0:last_layer]
        )

        # Dont calculate gradient when just use VGG19 as feature extractor.
        self.requires_grad = requires_grad
        for p in self.feature_extractor.parameters():
            p.requires_grad_(self.requires_grad)

        self.use_input_norm = use_input_norm
        if self.use_input_norm:
            """
            The tips of Pytorch:
                All pre-trained models expect input images normalized in the same way,
                i.e. mini-batches of 3-channel RGB images of shape (3 x H x W),
                where H and W are expected to be at least 224.
                The images have to be loaded in to a range of [0, 1],
                and then normalized using mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225].
                Ref: https://pytorch.org/docs/1.2.0/torchvision/models.html#torchvision-models.
            """
            mean = torch.as_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.as_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            self.register_buffer("mean", mean)
            self.register_buffer("std", std)
        else:
            print(
                "Waring: "
                "Please normalize the input image "
                "if you are using the pre-trained model provided by Pytorch."
            )

    def forward(self, x):
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        return self.feature_extractor(x)


def gradient_penalty(critic, real_samples, fake_samples, lambda_gp=10.0):
    """
    Gradient penalty term for WGAN-GP.

    :param critic:
        Alias of Discriminator.
    :param real_samples:
        Real input samples.
        Here you can regard it as the real images.
    :param fake_samples:
        Fake input samples.
        Here you can regard it as the generated images.
    :param lambda_gp:
        The weight of gradient penalty term.
        Default: 10.
    """
    import torch.autograd

    assert real_samples.size() == fake_samples.size(), (
        f"The size of real_samples {real_samples.size()} and fake_samples {fake_samples.size()} are not equal."
    )

    # Get the batch size.
    batch_size = real_samples.size(0)

    # Random weight term for interpolation between real and fake samples.
    # The func new_tensor will return tensor with same dtype and device as real_data.
    alpha = torch.rand(size=(batch_size, 1, 1, 1), dtype=real_samples.dtype, device=real_samples.device)

    # Get random interpolation between real and fake samples
    interpolates = alpha * real_samples + (1.0 - alpha) * fake_samples
    interpolates.requires_grad_(True)

    d_interpolates = critic(interpolates)

    # Calculate the gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates).requires_grad_(False),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return lambda_gp * gp


if __name__ == '__main__':
    pass
