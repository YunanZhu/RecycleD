"""
The dataloader of super-resolution training set.
"""
import numpy as np
from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as transforms_f


class PairImageTrainSet(Dataset):
    def __init__(self,
                 path: str,
                 guide_file: str,
                 lr_patch_size: int = 48,
                 scale_factor: int = 4,
                 use_data_augmentation: bool = True):
        """
        Load super-resolution training set which has paired images (HR & LR).

        :param path:
            The path of image set.
        :param guide_file:
            A text file which has paired image info. Used to locate the image files.
        :param lr_patch_size:
            The size of cropped LR patch.
            Here LR patch has same height and width.
            Default: 48.
        :param scale_factor:
            Upscale factor.
            Default: 4.
        :param use_data_augmentation:
            If set to True, conduct data augmentation.
            Default: True.
        """
        super(PairImageTrainSet, self).__init__()

        self.path = path

        self.lr_patch_size = lr_patch_size
        self.scale_factor = scale_factor

        pairs = [line.rstrip() for line in open(guide_file, mode='r')]
        self.image_pairs = [
            one_pair.split(',') for one_pair in pairs
        ]  # It's a list containing many HR+LR pairs, and each pair is a list of 2 filenames.

        self.use_data_augmentation = use_data_augmentation

    def __getitem__(self, index):
        hr_image, lr_image = tuple(self.image_pairs[index])
        hr_image = Image.open(self.path + hr_image)
        lr_image = Image.open(self.path + lr_image)

        hr_w, hr_h = hr_image.size
        lr_w, lr_h = lr_image.size
        assert hr_w == self.scale_factor * lr_w and hr_h == self.scale_factor * lr_h, (
            f"The size difference of HR and LR images are not X{self.scale_factor} scale factor."
        )

        # Set the upper left point of the patch to crop.
        point = (
            np.random.randint(0, lr_h - self.lr_patch_size + 1),
            np.random.randint(0, lr_w - self.lr_patch_size + 1)
        )

        # Crop HR and LR patches.
        lr_patch = transforms_f.crop(
            lr_image,
            point[0], point[1],
            self.lr_patch_size, self.lr_patch_size
        )
        hr_patch = transforms_f.crop(
            hr_image,
            self.scale_factor * point[0], self.scale_factor * point[1],
            self.scale_factor * self.lr_patch_size, self.scale_factor * self.lr_patch_size
        )

        # Augment data.
        if self.use_data_augmentation:
            if np.random.random() < 0.5:  # random rotations
                if np.random.random() < 0.5:
                    # counter-clockwise 90 degrees
                    lr_patch = transforms_f.rotate(lr_patch, angle=90)
                    hr_patch = transforms_f.rotate(hr_patch, angle=90)
                else:
                    # clockwise 90 degrees
                    lr_patch = transforms_f.rotate(lr_patch, angle=270)
                    hr_patch = transforms_f.rotate(hr_patch, angle=270)
            if np.random.random() < 0.5:  # random horizontal flips
                lr_patch = transforms_f.hflip(lr_patch)
                hr_patch = transforms_f.hflip(hr_patch)

        # From pil image to pytorch tensor.
        lr_patch = transforms_f.to_tensor(lr_patch)
        hr_patch = transforms_f.to_tensor(hr_patch)

        return {"LR": lr_patch, "HR": hr_patch}

    def __len__(self):
        return len(self.image_pairs)


class SingleImageTrainSet(Dataset):
    def __init__(self,
                 path: str,
                 guide_file: str,
                 hr_patch_size: int = 192,
                 scale_factor: int = 4,
                 use_data_augmentation: bool = True):
        """
        Load super-resolution training set which has many HR images but no LR image.

        :param path:
            The path of image set.
        :param guide_file:
            A text file which has HR image info. Used to locate the image files.
        :param hr_patch_size:
            The size of cropped HR patch.
            HR patch has same height and width.
            Default: 192.
        :param scale_factor:
            Upscale factor.
            Default: 4.
        :param use_data_augmentation:
            If set to True, conduct data augmentation.
            Default: True.
        """
        super(SingleImageTrainSet, self).__init__()

        self.path = path

        self.hr_patch_size = hr_patch_size
        self.scale_factor = scale_factor

        self.images = [line.rstrip() for line in open(guide_file, mode='r')]

        self.use_data_augmentation = use_data_augmentation

    def __getitem__(self, index):
        from utils import matlab_functions

        hr_image = self.images[index]
        hr_image = cv2.imread(self.path + hr_image, cv2.IMREAD_COLOR)
        hr_image = cv2.cvtColor(
            hr_image,
            code=cv2.COLOR_BGR2RGB
        )  # Convert from BGR to RGB, unified with the Pillow library.

        hr_h, hr_w, hr_c = hr_image.shape

        # Get the upper left point of the patch.
        point = (
            np.random.randint(0, hr_h - self.hr_patch_size + 1),
            np.random.randint(0, hr_w - self.hr_patch_size + 1)
        )

        # Crop HR patch and generate LR patch.
        hr_patch = hr_image[
                   point[0]:(point[0] + self.hr_patch_size),
                   point[1]:(point[1] + self.hr_patch_size),
                   :]
        lr_patch = matlab_functions.imresize(
            hr_patch,
            scale=1.0 / self.scale_factor,
            antialiasing=True
        )  # Generate LR patch by bicubic down-sampling (following MATLAB).
        lr_patch = np.round(lr_patch).astype(hr_patch.dtype)

        # Augment data.
        if self.use_data_augmentation:
            if np.random.random() < 0.5:  # random rotations
                if np.random.random() < 0.5:
                    # counter-clockwise 90 degrees
                    lr_patch = cv2.rotate(lr_patch, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
                    hr_patch = cv2.rotate(hr_patch, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
                else:
                    # clockwise 90 degrees (rotateCode=0 in cv2)
                    lr_patch = cv2.rotate(lr_patch, rotateCode=cv2.ROTATE_90_CLOCKWISE)
                    hr_patch = cv2.rotate(hr_patch, rotateCode=cv2.ROTATE_90_CLOCKWISE)
            if np.random.random() < 0.5:  # random horizontal flips
                # horizontal flips (flipCode=1 in cv2)
                lr_patch = cv2.flip(lr_patch, flipCode=1)
                hr_patch = cv2.flip(hr_patch, flipCode=1)

        # From numpy array to pytorch tensor.
        lr_patch = transforms_f.to_tensor(lr_patch)
        hr_patch = transforms_f.to_tensor(hr_patch)

        return {"LR": lr_patch, "HR": hr_patch}

    def __len__(self):
        return len(self.images)


def get_dataloader(choice: str,
                   path: str,
                   guide_file: str,
                   lr_patch_size: int,
                   scale_factor: int,
                   use_data_augmentation: bool,
                   batch_size: int,
                   shuffle: bool = True,
                   num_workers: int = 0):
    """
    Return a dataloader for training.

    :param choice:
        Use paired images training set or single image training set.
        Support: "Pair", "Single".
    :param path:
        The path of image set.
    :param guide_file:
        A text file which has training set image info. Used to locate the image files.
    :param lr_patch_size:
        The size of cropped LR patch.
        The patch has same height and width.
    :param scale_factor:
        Upscale factor.
    :param use_data_augmentation:
        If set to True, conduct data augmentation.
    :param batch_size:
        The size of one batch.
    :param shuffle:
        Indicate if shuffle when load samples.
    :param num_workers:
        How many subprocesses to use for data loading.
    """
    from torch.utils.data import DataLoader

    if choice == "Pair":
        train_set = PairImageTrainSet(
            path, guide_file,
            lr_patch_size=lr_patch_size,
            scale_factor=scale_factor,
            use_data_augmentation=use_data_augmentation
        )
    elif choice == "Single":
        train_set = SingleImageTrainSet(
            path, guide_file,
            hr_patch_size=lr_patch_size * scale_factor,
            scale_factor=scale_factor,
            use_data_augmentation=use_data_augmentation
        )
    else:
        raise NotImplementedError(f"The choice = {choice} is illegal.")

    return DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )


if __name__ == '__main__':
    pass
