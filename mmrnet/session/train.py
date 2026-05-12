import os
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import pytorch_lightning as pl
import torch
from .wrapper import ModelWrapper
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

def plt_model_load(model, checkpoint):
    state_dict = torch.load(checkpoint)['state_dict']
    model.load_state_dict(state_dict, strict=False)
    return model

def train(
        model,
        train_loader, val_loader,
        optimizer, 
        learning_rate, 
        weight_decay,
        plt_trainer_args, 
        save_path='./lightning_logs/',
        log_params=None,
        early_stop=False,
        load_path=None
        ):

    plt_model = ModelWrapper(
        model,
        dataset=train_loader.dataset,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epochs=plt_trainer_args['max_epochs'],
        optimizer=optimizer,
    )

    if load_path is not None:
        if load_path.endswith(".ckpt"):
            checkpoint = load_path
        else:
            if load_path.endswith("/"):
                checkpoint = load_path + "best.ckpt"
            else:
                raise ValueError(
                    "if it is a directory, it must end with /; if it is a file, it must end with .ckpt")
        plt_model = plt_model_load(plt_model, checkpoint)
        print(f"Loaded model from {checkpoint}")
    
    metric = f'val_{plt_model.metric_name}'
    if 'mle' in metric:
        mode = 'min'
    elif 'acc' in metric:
        mode = 'max'
    else:
        raise ValueError(f'Unknown metric {metric}')
    
    if 'tuning' in save_path:
        logger = TensorBoardLogger(save_dir=save_path, name='', version='')
        if "SLURM_JOB_ID" in os.environ:
            refresh_rate = 0
        else:
            refresh_rate = 5
    else:
        logger = TensorBoardLogger(save_dir=save_path, name='')
        refresh_rate = 5

    plt_trainer_args['logger'] = logger
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        monitor=metric,
        mode=mode,
        filename="best",
        dirpath=logger.log_dir,
        save_last=False,
    )
    plt_trainer_args['callbacks'] = [checkpoint_callback]
    print(f"Logging and saving to {logger.log_dir}")
    
    if early_stop:
        early_stop_callback = EarlyStopping(
            monitor=metric,
            patience=max(5, plt_trainer_args['max_epochs'] // 10),
            verbose=True,
            mode=mode,
        )
        plt_trainer_args['callbacks'].append(early_stop_callback)

    progress_bar = TQDMProgressBar(refresh_rate=refresh_rate)
    plt_trainer_args['callbacks'].append(progress_bar)
    trainer = pl.Trainer(**plt_trainer_args)

    # Log hyperparameters using the trainer's logger
    if log_params:
        trainer.logger.log_hyperparams(log_params)

    trainer.fit(plt_model, train_loader, val_loader)
    return plt_model.best_val_loss
