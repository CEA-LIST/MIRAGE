import os
import sys
import random
import argparse
import logging
from datetime import datetime

import torch
import numpy as np
import toml
import optuna
import pytorch_lightning as pl

from .session import train, test
from .models import model_map 
from .dataset import get_dataset
from .session.wrapper import ModelWrapper  # Ensure ModelWrapper is imported

logging.getLogger().setLevel(logging.INFO)

class Main:
    arguments = {
        ('action', ): {'type': str, 'help': 'Name of the action to perform.'},
        ('dataset', ): {'type': str, 'help': 'Name of the dataset.'},
        ('model', ): {'type': str, 'help': 'Name of the model.'},

        # checkpoint
        ('-load', '--load_name'): {
            'type': str, 'default': None,
            'help': 'Name of the saved model to restore.'
        },
        ('-save', '--save_name'): {
            'type': str, 'default': './lightning_logs/',
            'help': 'Directory to save the model checkpoints and logs.',
        },

        # common training args
        ('-opt', '--optimizer'): {
            'type': str, 'default': 'adam', 'help': 'Pick an optimizer.',
        },
        ('-lr', '--learning_rate'): {
            'type': float, 'default': 3e-5, 'help': 'Initial learning rate.',
        },
        ('-m', '--max_epochs'): {
            'type': int, 'default': 100,
            'help': 'Maximum number of epochs for training.',
        },
        ('-b', '--batch_size'): {
            'type': int, 'default': 128,
            'help': 'Batch size for training and evaluation.',
        },
        ('-wd', '--weight_decay'): {
            'type': float, 'default': 1e-5, 'help': 'Weight decay for optimizer regularization.'
        },

        # debug control
        ('-d', '--debug'): {
            'action': 'store_true', 'help': 'Verbose debug',
        },
        ('-seed', '--seed'): {
            'type': int, 'default': 0, 'help': 'Number of steps for model optimisation',
        },

        # cpu gpu setup for lightning
        ('-w', '--num_workers'): {
            # multiprocessing fail with too many works
            'type': int, 'default': 0, 'help': 'Number of CPU workers.',
        },
        ('-n', '--num_devices'): {
            'type': int, 'default': 1, 'help': 'Number of GPU devices.',
        },
        ('-a', '--accelerator'): {
            'type': str, 'default': None, 'help': 'Accelerator style.',
        },
        ('-s', '--strategy'): {
            'type': str, 'default': None, 'help': 'Strategy style, e.g. ddp.',
        },
        # dataset related
        ('-config', '--config'): {
            'type': str, 'default': None, 'help': 'Config file.',
        },
        ('-d_seed', '--dataset_seed'): {
            'type': int, 'default': None, 'help': 'Dataset seed.',
        },
        ('-d_stacks', '--dataset_stacks'): {
            'type': int, 'default': None, 'help': 'Number of mmPoints to stack.',
        },
        ('-d_zero_padding', '--dataset_zero_padding'): {
            'type': str, 'default': None, 'help': 'Zero padding styles.',
        },
        ('-d_max_points', '--dataset_max_points'): {
            'type': int, 'default': None, 'help': 'Point cloud population.',
        },
        ('-d_processed_data', '--dataset_processed_data'): {
            'type': str, 'default': None, 'help': 'Processed data file.',
        },
        ('-d_forced_rewrite', '--dataset_forced_rewrite'): {
            'action': 'store_true', 'help': 'Force to rewrite the processed data.',
        },
        ('-v', '--visualize'): {
            'action': 'store_true', 'help': 'Visualize test result as mp4.',
        },
        # tuner args
        ('-n_trials', '--n_trials'): {
            'type': int, 'default': 100, 'help': 'Number of trails to run.',
        },
        # custom args
        ('-d_include_time', '--dataset_include_time'): {
            'type': bool, 'default': None, 'help': 'Add frame sequence number as fourth dimension.',
        },
        ('-det', '--deterministic'): {
            'type': bool, 'default': False, 'help': 'Run the training in deterministic mode to improve reproducibility.'
        },
        ('-early_stop', '--early_stop'): {
            'action': 'store_true', 'help': 'Use early stopping during training.',
        },
        ('-d_data_augmentation', '--dataset_data_augmentation'): {
            'type': str, 'default': None, 'help': 'Define the data augmentation for the dataset.',
        },
        ('-d_kfold', '--dataset_kfold'): {
            'type': int, 'default': None, 'help': 'kfold validation ID.',
        },
        ('-d_input_features', '--dataset_input_features'): {
            'type': str, 'default': None, 'help': 'dataset input features (variable X).',
        },
        ('-d_posconv_features', '--dataset_posconv_features'): {
            'type': str, 'default': None, 'help': 'dataset edge features (variable E in pointnet, not used in CNN and Transformer).',
        },
        ('-d_posgraph_features', '--dataset_posgraph_features'): {
            'type': str, 'default': None, 'help': 'dataset distance computation features (variable D in pointnet, not used in CNN and Transformer).',
        },
        ('-m_t_scale', '--model_t_scale'): {
            'type': float, 'default': None, 'help': 'time scaling factor (only used in pointnet when dataset_posgraph_features includes time).',
        },
        ('-m_fps_ratios', '--model_fps_ratios'): {
            'type': str, 'default': None, 'help': 'fps ratio parameter (only used in pointnet).',
        },
        ('-m_radius', '--model_radius'): {
            'type': str, 'default': None, 'help': 'radius parameter (only used in pointnet).',
        },
    }

    def __init__(self):
        super().__init__()
        a = self.parse()
        if a.debug:
            sys.excepthook = self._excepthook
        # seeding
        random.seed(a.seed)
        torch.manual_seed(a.seed)
        np.random.seed(a.seed)
        torch.cuda.manual_seed_all(a.seed)
        
        # Added seed setting to address reproducibility problem
        if a.deterministic:
            print("Running in deterministic mode.")
            pl.seed_everything(a.seed, workers=True)

        self.a = a

    def parse(self):
        p = argparse.ArgumentParser(description='Millimeter Wave Radar Dataset.')
        for k, v in self.arguments.items():
            p.add_argument(*k, **v)
        p = p.parse_args()
        p = self.post_parse(p)
        return p
    
    def post_parse(self, p):
        # Load from a TOML config file
        if p.config is not None:
            if not p.config.endswith('.toml'):
                raise ValueError('Config file must be a TOML file.')
            with open(p.config, 'r') as f:
                config = toml.load(f)

            # Track explicitly set CLI arguments
            cli_args = vars(p)
            explicitly_set = set()
            for k, v in self.arguments.items():
                arg_name = k[1].lstrip('-').replace('-', '_') if len(k) > 1 else k[0].lstrip('-').replace('-', '_')  # Convert to attribute name
                if arg_name in cli_args and cli_args[arg_name] != v.get('default'):
                    explicitly_set.add(arg_name)

            # Flatten TOML sections and apply values
            def flatten_dict(d, parent_key=''):
                items = []
                for k, v in d.items():
                    new_key = f"{parent_key}_{k}" if parent_key else k
                    if isinstance(v, dict):
                        items.extend(flatten_dict(v, new_key).items())
                    else:
                        items.append((new_key, v))
                return dict(items)

            flat_config = flatten_dict(config)

            # Apply TOML values only if the argument was not explicitly set
            for k, v in flat_config.items():
                if k in cli_args:  # Check if the key exists in the parsed CLI arguments
                    if k not in explicitly_set:  # Only use the TOML value if not explicitly set
                        setattr(p, k, v)
                    else:
                        print(
                            f'[Config setup] Config value {v} is not used for {k}, '
                            f'command line has a higher priority and sets it to {cli_args[k]}')
                else:
                    raise ValueError(f'Unknown config key {k} in the TOML file.')

        return p

    def _excepthook(self, etype, evalue, etb):
        from IPython.core import ultratb
        ultratb.FormattedTB()(etype, evalue, etb)
        for exc in [KeyboardInterrupt, FileNotFoundError]:
            if issubclass(etype, exc):
                sys.exit(-1)
        import ipdb
        ipdb.post_mortem(etb)

    def run(self):
        try:
            action = getattr(self, f'cli_{self.a.action.replace("-", "_")}')
        except AttributeError:
            callables = [n[4:] for n in dir(self) if n.startswith('cli_')]
            logging.error(
                f'Unkown action {self.a.action!r}, '
                f'accepts: {", ".join(callables)}.')
        return action()

    def setup_model_and_data(self, a, dataset_custom_args=None):
        # get dataset
        logging.info(f'Loading dataset {a.dataset!r}...')

        dataset_stacks = None if dataset_custom_args is None else dataset_custom_args.get('stacks', None)
        my_stacks = a.dataset_stacks if dataset_stacks is None else dataset_stacks

        path = None

        if "cnn" in a.model:
            model_type = "voxel"
            if "lstm" in a.model:
                model_type = "voxellstm"
        elif a.dataset_zero_padding in ['per_data_point', 'data_point']:
            model_type = "padded"
        else:
            model_type = "graph"

        mmr_dataset_config = {
            'seed': a.dataset_seed,
            'stacks': my_stacks,
            'zero_padding': a.dataset_zero_padding,
            'processed_data': a.dataset_processed_data if path is None else path,
            'forced_rewrite': a.dataset_forced_rewrite,
            'max_points': a.dataset_max_points,
            'include_time' : a.dataset_include_time,
            'data_augmentation': a.dataset_data_augmentation if dataset_custom_args is None else dataset_custom_args.get('data_augmentation', None),
            'input_features': a.dataset_input_features,
            'posconv_features': a.dataset_posconv_features,
            'posgraph_features': a.dataset_posgraph_features,
            'kfold': a.dataset_kfold,
            'model_type': model_type,
        }
        train_loader, val_loader, test_loader, dataset_info = get_dataset(
            name=a.dataset, 
            batch_size=a.batch_size, 
            workers=a.num_workers,
            mmr_dataset_config=mmr_dataset_config)
        logging.info(f'Loaded dataset {a.dataset!r}.')

        mmr_model_config = {
            'num_classes': dataset_info.get('num_classes'),
            'num_features': dataset_info.get('num_features', 3),
            'num_pos_features': dataset_info.get('num_pos_features', 3),
            't_scale': a.model_t_scale if a.model_t_scale != 'None' else None,
            'fps_ratios': a.model_fps_ratios,
            'radius': a.model_radius,
        }

        # get model
        model_cls = model_map[a.model]
        model = model_cls(info=mmr_model_config)
        return model, train_loader, val_loader, test_loader, dataset_info

    def cli_train(self, dataset_custom_args=None, train_custom_args=None):
        a = self.a
        if not a.save_name:
            logging.error('--save_name not specified.')
            sys.exit(1)

        if dataset_custom_args is not None:
            for k, v in dataset_custom_args.items():
                setattr(a, k, v)

        if train_custom_args is not None:
            for k, v in train_custom_args.items():
                setattr(a, k, v)

        model, train_loader, val_loader, test_loader, dataset_info = self.setup_model_and_data(
            a, dataset_custom_args=dataset_custom_args)
        
        for k, v in vars(model).items():
            k = f"model_{k}"
            if k in vars(a):
                setattr(a, k, v)

        if a.strategy == 'ddp':
            a.strategy = 'ddp_find_unused_parameters_false'
        plt_trainer_args = {
            'max_epochs': a.max_epochs, 
            'devices': a.num_devices,
            'accelerator': a.accelerator, 
            'strategy': a.strategy,
            'fast_dev_run': a.debug, 
            'deterministic': "warn" if a.deterministic else None,   # Make results reproducible
        }
        
        if a.load_name:
            load_path = a.load_name if a.load_name.endswith(".ckpt") else 'lightning_logs/' + a.load_name + '/'
        else:
            load_path = None

        train_params = {
            'model': model,
            'load_path': load_path,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'optimizer': a.optimizer,
            'learning_rate': a.learning_rate,
            "plt_trainer_args": plt_trainer_args,
            "weight_decay": a.weight_decay,
            "save_path": a.save_name,
            'early_stop': a.early_stop,
        }        

        train_params["log_params"] = vars(a)
        print(train_params["log_params"])

        loss = train(**train_params)
        return loss

    def cli_test(self, dataset_custom_args=None):
        a = self.a

        model, train_loader, val_loader, test_loader, dataset_info = self.setup_model_and_data(
            a, dataset_custom_args=dataset_custom_args)
        
        load_path = a.load_name if a.load_name.endswith(".ckpt") else a.load_name + '/'

        plt_trainer_args = {
            'devices': a.num_devices,
            'accelerator': a.accelerator, 'strategy': a.strategy,}
        test_params = {
            'model': model,
            'test_loader': test_loader,
            'plt_trainer_args': plt_trainer_args,
            'load_path': load_path,
        }
        test(**test_params)
    cli_eval = cli_test

    def cli_evalnoisy(self, dataset_custom_args=None):
        a = self.a
        load_path = a.load_name if a.load_name.endswith(".ckpt") else a.load_name + '/'

        plt_trainer_args = {
            'devices': a.num_devices,
            'accelerator': a.accelerator, 'strategy': a.strategy,}

        ## clean eval
        if "evalnoisy" in a.dataset_processed_data:
            noisy = "noisy"
        elif "evalclean" in a.dataset_processed_data:
            noisy = "clean"
        else:
            raise ValueError("not implemented process type: processed_data must be either evalnoisy or evalclean")
        
        model, train_loader, val_loader, test_loader, dataset_info = self.setup_model_and_data(
            a, dataset_custom_args=dataset_custom_args)
        
        test_params = {
            'model': model,
            'test_loader': val_loader, ## eval on validation set
            'plt_trainer_args': plt_trainer_args,
            'load_path': load_path,
            'noisy': noisy,
        }
        test(**test_params)



    def cli_gnn_features_ablation(self):
        # Define the hyperparameter search space
        if "pointnet_tiny" in self.a.model:
            search_space = {
                "input_features": [
                    '[\"xyz\"]',
                    '[\"xyz\", \"time\"]',
                    '[\"xyz\", \"velocity\"]',
                    '[\"xyz\", \"intensity\"]',
                ],
                "posconv_features": [
                    '[\"xyz\"]',
                    '[\"xyz\", \"time\"]',
                    '[\"xyz\", \"velocity\"]',
                    '[\"xyz\", \"intensity\"]',
                ],
                "posgraph_features": [
                    '[\"xyz\"]'
                ],
            }

        def objective(trial):
            # Extract hyperparameters from the trial
            self.a.dataset_input_features = trial.suggest_categorical("input_features", search_space["input_features"])
            self.a.dataset_posconv_features = trial.suggest_categorical("posconv_features", search_space["posconv_features"])
            self.a.dataset_posgraph_features = trial.suggest_categorical("posgraph_features", search_space["posgraph_features"])

            ## to do feature by feature ablation, removes trials mixing feature types
            if self.a.dataset_input_features == '[\"xyz\", \"time\"]':
                if self.a.dataset_posconv_features == '[\"xyz\", \"velocity\"]' or self.a.dataset_posconv_features == '[\"xyz\", \"intensity\"]':
                    print("Skipping trial input_features: {input_features}, posconv_features: {posconv_features}")
                    return float("inf")
            elif self.a.dataset_input_features == '[\"xyz\", \"velocity\"]':
                if self.a.dataset_posconv_features == '[\"xyz\", \"time\"]' or self.a.dataset_posconv_features == '[\"xyz\", \"intensity\"]':
                    print("Skipping trial input_features: {input_features}, posconv_features: {posconv_features}")
                    return float("inf")
            elif self.a.dataset_input_features == '[\"xyz\", \"intensity\"]':
                if self.a.dataset_posconv_features == '[\"xyz\", \"time\"]' or self.a.dataset_posconv_features == '[\"xyz\", \"velocity\"]':
                    print("Skipping trial input_features: {input_features}, posconv_features: {posconv_features}")
                    return float("inf")

            base_trial_name = "version"
            trial_index = 0
            while os.path.exists(os.path.join(self.save_name, f"{base_trial_name}_{trial_index}")):
                trial_index += 1
            self.a.save_name = os.path.join(self.save_name, f"{base_trial_name}_{trial_index}")
            
            logging.info(f"Trial {trial.number}: input_features={self.a.dataset_input_features}, posconv_features={ self.a.dataset_posconv_features}, posgraph_features={self.a.dataset_posgraph_features}, fps_ratios={self.a.model_fps_ratios}, radius={self.a.model_radius}, t_scale={self.a.model_t_scale}, data_augmentation={self.a.dataset_data_augmentation}")
            
            a = self.a
            
            # Run training and return validation loss
            val_loss = self.cli_train()
            self.a.load_name = a.save_name
            self.cli_test()
            self.a.load_name = None  # Reset load_name for the next trial
            return val_loss

        # Ensure unique tuning folder name
        base_tuning_name = "tuning"
        tuning_index = 0
        while os.path.exists(os.path.join(self.a.save_name, f"{base_tuning_name}_{tuning_index}")):
            tuning_index += 1
        self.save_name = os.path.join(self.a.save_name, f"{base_tuning_name}_{tuning_index}")

        # Use GridSampler for exhaustive search
        sampler = optuna.samplers.GridSampler(search_space)
        pruner = optuna.pruners.NopPruner()
        # Create the study with no pruning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study = optuna.create_study(
            study_name=f"hyperparameter_tuning_{timestamp}",
            storage=f"sqlite:///tuning_database.db",
            direction="minimize",
            load_if_exists=True, 
            sampler=sampler, 
            pruner=pruner)

        # Optimize the study
        study.optimize(objective)

        # Print the best trial
        print(study.best_trial)


    def cli_models_features_ablation(self):
        # Define the hyperparameter search space

        if "pointnet_tiny" in self.a.model:
            search_space = {
                "input_features": [
                    '[\"xyz\"]'
                ],
                "posconv_features": [
                    '[\"xyz\"]',
                    '[\"xyz\", \"time\"]',
                    '[\"xyz\", \"velocity\"]',
                    '[\"xyz\", \"intensity\"]',
                    '[\"xyz\", \"time\", \"velocity\", \"intensity\"]',
                ],
                "posgraph_features": [
                    '[\"xyz\"]'
                ],
            }

        elif "transformer_tiny" in self.a.model:
            search_space = {
                "input_features": [
                    '[\"xyz\"]',
                    '[\"xyz\", \"time\"]',
                    '[\"xyz\", \"velocity\"]',
                    '[\"xyz\", \"intensity\"]',
                    '[\"xyz\", \"time\", \"velocity\", \"intensity\"]',
                ],
                "posconv_features": [
                    '[\"xyz\"]',
                ],
                "posgraph_features": [
                    '[\"xyz\"]',
                ],
            }

        elif ("tdcnnlstm" in self.a.model or "cnn_tiny" in self.a.model):
            search_space = {
                "input_features": [
                    '[\"xyz\"]',
                    '[\"xyz\", \"velocity\"]',
                    '[\"xyz\", \"time\"]',
                    '[\"xyz\", \"intensity\"]',
                    '[\"xyz\", \"time\", \"velocity\", \"intensity\"]',
                ],
                "posconv_features": [
                    '[\"xyz\"]',
                ],
                "posgraph_features": [
                    '[\"xyz\"]',
                ],
            }

        def objective(trial):
            # Extract hyperparameters from the trial
            self.a.dataset_input_features = trial.suggest_categorical("input_features", search_space["input_features"])
            self.a.dataset_posconv_features = trial.suggest_categorical("posconv_features", search_space["posconv_features"])
            self.a.dataset_posgraph_features = trial.suggest_categorical("posgraph_features", search_space["posgraph_features"])

            base_trial_name = "version"
            trial_index = 0
            while os.path.exists(os.path.join(self.save_name, f"{base_trial_name}_{trial_index}")):
                trial_index += 1
            self.a.save_name = os.path.join(self.save_name, f"{base_trial_name}_{trial_index}")
            
            logging.info(f"Trial {trial.number}: input_features={self.a.dataset_input_features}, posconv_features={ self.a.dataset_posconv_features}, posgraph_features={self.a.dataset_posgraph_features}, fps_ratios={self.a.model_fps_ratios}, radius={self.a.model_radius}, t_scale={self.a.model_t_scale}, data_augmentation={self.a.dataset_data_augmentation}")
            
            a = self.a
            
            # Run training and return validation loss
            val_loss = self.cli_train()
            self.a.load_name = a.save_name.replace("./lightning_logs/", "")
            self.cli_test()
            self.a.load_name = None  # Reset load_name for the next trial
            return val_loss

        # Ensure unique tuning folder name
        base_tuning_name = "tuning"
        tuning_index = 0
        while os.path.exists(os.path.join(self.a.save_name, f"{base_tuning_name}_{tuning_index}")):
            tuning_index += 1
        self.save_name = os.path.join(self.a.save_name, f"{base_tuning_name}_{tuning_index}")

        # Use GridSampler for exhaustive search
        sampler = optuna.samplers.GridSampler(search_space)
        pruner = optuna.pruners.NopPruner()
        # Create the study with no pruning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study = optuna.create_study(
            study_name=f"hyperparameter_tuning_{timestamp}",
            storage=f"sqlite:///tuning_database.db",
            direction="minimize",
            load_if_exists=True, 
            sampler=sampler, 
            pruner=pruner)

        # Optimize the study
        study.optimize(objective)

        # Print the best trial
        print(study.best_trial)

    
    def cli_models_clean_training(self):
        # Define the hyperparameter search space

        if "pointnet_tiny" in self.a.model:
            search_space = {
                "input_features": [
                    '[\"xyz\"]'
                ],
                "posconv_features": [
                    '[\"xyz\"]',
                    '[\"xyz\", \"time\"]',
                    '[\"xyz\", \"velocity\"]',
                    '[\"xyz\", \"intensity\"]',
                    '[\"xyz\", \"time\", \"velocity\", \"intensity\"]',
                ],
                "posgraph_features": [
                    '[\"xyz\"]'
                ],
            }

        elif "cnn_tiny" in self.a.model:
            if "mirage" in self.a.dataset:
                search_space = {
                    "input_features": [
                        '[\"xyz\", \"velocity\"]',
                    ],
                    "posconv_features": [
                        '[\"xyz\"]',
                    ],
                    "posgraph_features": [
                        '[\"xyz\"]',
                    ],
            }
            elif "radhar" in self.a.dataset:
                search_space = {
                    "input_features": [
                        '[\"xyz\", \"intensity\"]',
                    ],
                    "posconv_features": [
                        '[\"xyz\"]',
                    ],
                    "posgraph_features": [
                        '[\"xyz\"]',
                    ],
            }

        elif "transformer_tiny" in self.a.model:
            search_space = {
                "input_features": [
                    '[\"xyz\", \"time\", \"velocity\", \"intensity\"]'
                ],
                "posconv_features": [
                    '[\"xyz\"]'
                ],
                "posgraph_features": [
                    '[\"xyz\"]'
                ],
            }

        def objective(trial):
            # Extract hyperparameters from the trial
            self.a.dataset_input_features = trial.suggest_categorical("input_features", search_space["input_features"])
            self.a.dataset_posconv_features = trial.suggest_categorical("posconv_features", search_space["posconv_features"])
            self.a.dataset_posgraph_features = trial.suggest_categorical("posgraph_features", search_space["posgraph_features"])

            base_trial_name = "version"
            trial_index = 0
            while os.path.exists(os.path.join(self.save_name, f"{base_trial_name}_{trial_index}")):
                trial_index += 1
            self.a.save_name = os.path.join(self.save_name, f"{base_trial_name}_{trial_index}")
            
            logging.info(f"Trial {trial.number}: input_features={self.a.dataset_input_features}, posconv_features={ self.a.dataset_posconv_features}, posgraph_features={self.a.dataset_posgraph_features}, fps_ratios={self.a.model_fps_ratios}, radius={self.a.model_radius}, t_scale={self.a.model_t_scale}, data_augmentation={self.a.dataset_data_augmentation}")
            
            a = self.a
            
            # Run training and return validation loss
            val_loss = self.cli_train()
            self.a.load_name = a.save_name.replace("./lightning_logs/", "")
            self.cli_test()
            self.a.load_name = None  # Reset load_name for the next trial
            return val_loss

        # Ensure unique tuning folder name
        base_tuning_name = "tuning"
        tuning_index = 0
        while os.path.exists(os.path.join(self.a.save_name, f"{base_tuning_name}_{tuning_index}")):
            tuning_index += 1
        self.save_name = os.path.join(self.a.save_name, f"{base_tuning_name}_{tuning_index}")

        # Use GridSampler for exhaustive search
        sampler = optuna.samplers.GridSampler(search_space)
        pruner = optuna.pruners.NopPruner()
        # Create the study with no pruning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study = optuna.create_study(
            study_name=f"hyperparameter_tuning_{timestamp}",
            storage=f"sqlite:///tuning_database.db",
            direction="minimize",
            load_if_exists=True, 
            sampler=sampler, 
            pruner=pruner)

        # Optimize the study
        study.optimize(objective)

        # Print the best trial
        print(study.best_trial)


    def cli_gnn_tscale_ablation(self):
        # Define the hyperparameter search space

        if "pointnet_tiny" in self.a.model:
            if "mirage" in self.a.dataset:
                search_space = {
                    "input_features": [
                        '[\"xyz\"]'
                    ],
                    "posconv_features": [
                        '[\"xyz\", \"time\"]'
                    ],
                    "posgraph_features": [
                        '[\"xyz\", \"time\"]'
                    ],
                    "t_scale": [0.25, 0.5, 1.0, 2.0]
                }
            elif "radhar" in self.a.dataset:
                search_space = {
                    "input_features": [
                        '[\"xyz\"]'
                    ],
                    "posconv_features": [
                        '[\"xyz\", \"velocity\"]'
                    ],
                    "posgraph_features": [
                        '[\"xyz\", \"time\"]'
                    ],
                    "t_scale": [0.13, 0.25, 0.5, 1.0]
                }

        def objective(trial):
            # Extract hyperparameters from the trial
            self.a.dataset_input_features = trial.suggest_categorical("input_features", search_space["input_features"])
            self.a.dataset_posconv_features = trial.suggest_categorical("posconv_features", search_space["posconv_features"])
            self.a.dataset_posgraph_features = trial.suggest_categorical("posgraph_features", search_space["posgraph_features"])
            self.a.model_t_scale = trial.suggest_categorical("t_scale", search_space["t_scale"])


            base_trial_name = "version"
            trial_index = 0
            while os.path.exists(os.path.join(self.save_name, f"{base_trial_name}_{trial_index}")):
                trial_index += 1
            self.a.save_name = os.path.join(self.save_name, f"{base_trial_name}_{trial_index}")
            
            logging.info(f"Trial {trial.number}: input_features={self.a.dataset_input_features}, posconv_features={ self.a.dataset_posconv_features}, posgraph_features={self.a.dataset_posgraph_features}, fps_ratios={self.a.model_fps_ratios}, radius={self.a.model_radius}, t_scale={self.a.model_t_scale}, data_augmentation={self.a.dataset_data_augmentation}")
            
            a = self.a
            
            # Run training and return validation loss
            val_loss = self.cli_train()
            self.a.load_name = a.save_name.replace("./lightning_logs/", "")
            self.cli_test()
            self.a.load_name = None  # Reset load_name for the next trial
            return val_loss

        # Ensure unique tuning folder name
        base_tuning_name = "tuning"
        tuning_index = 0
        while os.path.exists(os.path.join(self.a.save_name, f"{base_tuning_name}_{tuning_index}")):
            tuning_index += 1
        self.save_name = os.path.join(self.a.save_name, f"{base_tuning_name}_{tuning_index}")

        # Use GridSampler for exhaustive search
        sampler = optuna.samplers.GridSampler(search_space)
        pruner = optuna.pruners.NopPruner()
        # Create the study with no pruning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study = optuna.create_study(
            study_name=f"hyperparameter_tuning_{timestamp}",
            storage=f"sqlite:///tuning_database.db",
            direction="minimize",
            load_if_exists=True, 
            sampler=sampler, 
            pruner=pruner)

        # Optimize the study
        study.optimize(objective)

        # Print the best trial
        print(study.best_trial)


def main():
    Main().run()
