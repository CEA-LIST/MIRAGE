import os
import torch
import pytorch_lightning as pl
import numpy as np
import logging
from pytorch_lightning.loggers import TensorBoardLogger


from .wrapper import ModelWrapper


def get_checkpoint_file(checkpoint_dir):
    for file in os.listdir(checkpoint_dir):
        if file.endswith(".ckpt"):
            return file


def plt_model_load(model, checkpoint):
    state_dict = torch.load(checkpoint)['state_dict']
    model.load_state_dict(state_dict)
    return model


def test(model, test_loader, plt_trainer_args, load_path, noisy=None):
    plt_model = ModelWrapper(model)
    if load_path is not None:
        if load_path.endswith(".ckpt"):
            checkpoint = load_path
        else:
            if load_path.endswith("/"):
                checkpoint = load_path + "best.ckpt"
            else:
                raise ValueError(
                    "if it is a directory, if must end with /; if it is a file, it must end with .ckpt")
        plt_model = plt_model_load(plt_model, checkpoint)
        plt_model.eval()
        print(f"Loaded model from {checkpoint}")
    if noisy == "noisy":
        log_dir = os.path.join(load_path, "val_noisy")
    elif noisy == "clean":
        log_dir = os.path.join(load_path, "val_clean")
    else:
        log_dir = os.path.join(load_path, "test")
    logger = TensorBoardLogger(save_dir=log_dir, name='', version='')
    plt_trainer_args['logger'] = logger
    trainer = pl.Trainer(**plt_trainer_args)
    trainer.test(plt_model, test_loader)