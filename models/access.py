"""
Used to save model and load model.
"""
import torch
from torch import nn


def save_model(model, epoch: int, file: str):
    """
    Save model to file.

    :param model:
        The model to saved.
    :param epoch:
        The epoch number of the model at this moment.
    :param file:
        The file to save the model parameters.
    """
    assert isinstance(epoch, int), f"The epoch type should be {int} rather than {type(epoch)}."

    # Prepare the dict for saving.
    dict_to_save = {
        "epoch": epoch,
        "state": model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    }

    torch.save(dict_to_save, file)


def load_model(model, epoch: int, file: str):
    """
    Load file to model.

    :param model:
        The model to be assigned parameters.
    :param epoch:
        The designated epoch number.
        Used to check the correctness of loading.
    :param file:
        Saved model file.
    """
    assert isinstance(epoch, int), f"The type of epoch should be {int} rather than {type(epoch)}."

    loaded_dict = torch.load(file)

    # Check if it is an old model file (w/o epoch number).
    if "epoch" not in loaded_dict.keys():
        print("You load a state dict without epoch item.")
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(loaded_dict)
        else:
            model.load_state_dict(loaded_dict)

        return model
    else:
        # Check the correctness of epoch number.
        print(f"Please check: "
              f"You load a state dict with epoch = {loaded_dict['epoch']}, "
              f"and the epoch designated by yourself = {epoch}.")

        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(loaded_dict["state"])
        else:
            model.load_state_dict(loaded_dict["state"])

        return model
