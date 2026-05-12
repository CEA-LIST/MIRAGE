import logging
import os
import pytorch_lightning as pl
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Batch
from .visualize import plot_confusion_and_metrics

class ModelWrapper(pl.LightningModule):

    def __init__(
            self,
            model,
            dataset=None,
            learning_rate=5e-4,
            weight_decay=1e-5,
            epochs=200,
            optimizer=None):
        super().__init__()
        self.model = model
        self.dataset = dataset
        self.num_classes = self.model.num_classes
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss = torch.nn.CrossEntropyLoss()
        self.metric_name = 'acc'
        self.metric = acc
        self.epochs = epochs
        self.optimizer = optimizer
        self.train_losses = []
        self.val_losses = []

        self.best_val_loss = 10e9

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        log_dict = {}

        y_hat = self.forward(x)
        y_hard = y  # For accuracy tracking

        loss = self.loss(y_hat, y)

        # Store predictions and hard targets for metrics
        if not hasattr(self, 'train_predictions'):
            self.train_predictions = []
            self.train_ground_truths = []
        self.train_predictions.append(y_hat.detach().cpu())
        self.train_ground_truths.append(y_hard.detach().cpu())

        # Logging
        log_dict["loss"] = loss
        log_config = {
            "on_step": False,
            "on_epoch": True,
            "prog_bar": False,
            "logger": True,
            "sync_dist": True
        }

        # Determine batch size
        batch_size = len(x.batch) if isinstance(x, Batch) else len(x)

        # Log metrics with the determined batch size
        self.log_dict(log_dict, batch_size=batch_size, **log_config)

        return {"loss": loss}
    
    def on_train_epoch_end(self):
        # Concatenate all predictions and ground truths
        y_hat = torch.cat(self.train_predictions, dim=0)
        y = torch.cat(self.train_ground_truths, dim=0)

        # Clear stored predictions and ground truths
        self.train_predictions = []
        self.train_ground_truths = []

        metric = self.metric(y_hat, y)
        print("train", metric)
        log_dict = {
            f'{self.metric_name}': metric
        }
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

    def validation_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.forward(x)

        loss = self.loss(y_hat, y)

        # Store predictions and ground truths for the epoch
        if not hasattr(self, 'val_predictions'):
            self.val_predictions = []
            self.val_ground_truths = []
        self.val_predictions.append(y_hat.detach().cpu())
        self.val_ground_truths.append(y.detach().cpu())

        # Log metrics
        log_dict = {
            "val_loss": loss,
        }

        # Logging configuration
        log_config = {
            "on_step": False,
            "on_epoch": True,
            "prog_bar": True,
            "logger": True,
            "sync_dist": True
        }

        # Determine batch size
        batch_size = len(x.batch) if isinstance(x, Batch) else len(x)

        # Log metrics with the determined batch size
        self.log_dict(log_dict, batch_size=batch_size, **log_config)
        
        self.val_losses.append(loss)

    def on_validation_epoch_end(self):
        if len(self.val_losses):
            val_loss = torch.stack(self.val_losses).mean()
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
            self.val_losses = []

        # Concatenate all predictions and ground truths
        y_hat = torch.cat(self.val_predictions, dim=0)
        y = torch.cat(self.val_ground_truths, dim=0)

        # Clear stored predictions and ground truths
        self.val_predictions = []
        self.val_ground_truths = []

        metric = self.metric(y_hat, y)
        log_dict = {
            f'val_{self.metric_name}': metric
        }
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        # Call plot_confusion_and_metrics at the end of the epoch
        if self.metric_name == 'acc':
            output_dir = os.path.join(self.trainer.log_dir, f"validation/epoch_{self.current_epoch}")
            os.makedirs(output_dir, exist_ok=True)
            plot_confusion_and_metrics(y_hat, y, output_dir, display=False)

    def test_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        y_hat = self.forward(x)

        loss = self.loss(y_hat, y)

        # Store predictions and ground truths for the epoch
        if not hasattr(self, 'test_predictions'):
            self.test_predictions = []
            self.test_ground_truths = []
        self.test_predictions.append(y_hat.detach().cpu())
        self.test_ground_truths.append(y.detach().cpu())

        log_dict = {
            "test_loss": loss,
        }

        # Determine batch size
        batch_size = len(x.batch) if isinstance(x, Batch) else len(x)

        # Log metrics with the determined batch size
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            batch_size=batch_size
        )
        return {"test_loss": loss}

    def on_test_epoch_end(self):
        # Concatenate all predictions and ground truths
        y_hat = torch.cat(self.test_predictions, dim=0)
        y = torch.cat(self.test_ground_truths, dim=0)

        # Clear stored predictions and ground truths
        self.test_predictions = []
        self.test_ground_truths = []

        metric = self.metric(y_hat, y)
        log_dict = {
            f'test_{self.metric_name}': metric
        }
        if self.metric_name == 'acc':
            # Compute top-3 accuracy
            top3 = torch.topk(y_hat, 3, dim=1)[1]
            top3_acc = (top3 == y.unsqueeze(-1)).float().sum() / len(y)
            log_dict[f'test_top3_{self.metric_name}'] = top3_acc
            plot_confusion_and_metrics(y_hat, y, output_dir=self.trainer.log_dir, display=True)
        
        self.log_dict(log_dict, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def predict_step(self, batch, batch_idx):
        x, y = batch[0], batch[1]
        return self(x), y

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            opt = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            scheduler = CosineAnnealingLR(opt, T_max=self.epochs, eta_min=1e-6)
        elif self.optimizer in ['sgd_warmup', 'sgd']:
            opt = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=0.9,
                weight_decay=self.weight_decay,
                nesterov=True)
            if self.optimizer == 'sgd':
                scheduler = CosineAnnealingLR(
                    opt, T_max=self.epochs, eta_min=0.0)
        return {
            "optimizer": opt,
            "lr_scheduler":  scheduler}

def mean_localization_error(x, y):
    dist = (x-y).pow(2).sum(-1).sqrt().mean()
    return dist

def acc(x, y):
    acc = (torch.argmax(x, axis=1) == y).float().sum()/x.shape[0]
    return acc
