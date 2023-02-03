import sys
import random
import argparse
import functools
import logging

import torch
import numpy as np
import toml

from .session import train, test
from .models import model_map 
from .dataset import get_dataset

logging.getLogger().setLevel(logging.INFO)

class Main:
    arguments = {
        ('action', ): {'type': str, 'help': 'Name of the action to perform.'},
        ('dataset', ): {'type': str, 'help': 'Name of the dataset.'},
        ('model', ): {'type': str, 'help': 'Name of the model.'},

        # checkpoint
        ('-load', '--load-name'): {
            'type': str, 'default': None,
            'help': 'Name of the saved model to restore.'
        },
        ('-save', '--save-name'): {
            'type': str, 'default': None,
            'help': 'Name of the saved model to save.'
        },

        # common training args
        ('-opt', '--optimizer'): {
            'type': str, 'default': 'adam', 'help': 'Pick an optimizer.',
        },
        ('-lr', '--learning-rate'): {
            'type': float, 'default': 1e-5, 'help': 'Initial learning rate.',
        },
        ('-m', '--max-epochs'): {
            'type': float, 'default': 100,
            'help': 'Maximum number of epochs for training.',
        },
        ('-b', '--batch-size'): {
            'type': int, 'default': 128,
            'help': 'Batch size for training and evaluation.',
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
            'type': str, 'default': 'ddp', 'help': 'Strategy style.',
        },

        # configs related
        ('-config', '--config'): {
            'type': str, 'default': None, 'help': 'config.',
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
        self.a = a

    def parse(self):
        p = argparse.ArgumentParser(description='Millimeter Wave Radar Dataset.')
        for k, v in self.arguments.items():
            p.add_argument(*k, **v)
        p = p.parse_args()
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

    def setup_model_and_data(self, a):
        # get dataset
        logging.info(f'Loading dataset {a.dataset!r}...')

        loader = get_dataset(
            name=a.dataset, 
            batch_size=a.batch_size, 
            workers=a.num_workers)
        logging.info(f'Loaded dataset {a.dataset!r}.')
        # get model
        model_cls = model_map[a.model]
        model = model_cls(info=dataset_info)
        return model, train_loader, val_loader, test_loader

    def cli_train(self):
        a = self.a
        if not a.save_name:
            logging.error('--save-name not specified.')

        model, train_loader, val_loader, test_loader = self.setup_model_and_data(a)
        plt_trainer_args = {
            'max_epochs': a.max_epochs, 'devices': a.num_devices,
            'accelerator': a.accelerator, 'strategy': a.strategy,
            'fast_dev_run': a.debug,}
        
        train_params = {
            'model': model,
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'optimizer': a.optimizer,
            'learning_rate': a.learning_rate,
            "plt_trainer_args": plt_trainer_args,
            "save_path": 'checkpoints/' + a.save_name,
        }
        train(**train_params)

    def cli_test(self):
        a = self.a

        model, _, _, test_loader = self.setup_model_and_data(a)
        plt_trainer_args = {
            'devices': a.num_devices,
            'accelerator': a.accelerator, 'strategy': a.strategy,}
        test_params = {
            'model': model,
            'test_loader': test_loader,
            'plt_trainer_args': plt_trainer_args,
            'load_path': a.load_name,
        }
        test(**test_params)
    cli_eval = cli_test
