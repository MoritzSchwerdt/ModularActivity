import os
from collections import defaultdict
from typing import Any

import jax
import numpy as np
import optax
from etils.etqdm import tqdm
from flax.training import train_state, checkpoints
from jax import random
from torch.utils.tensorboard import SummaryWriter

from config import CHECKPOINT_PATH


class TrainState(train_state.TrainState):
    # Batch statistics from BatchNorm
    batch_stats : Any
    # PRNGKey for augmentations
    rng : Any


class TrainerModule:

    def __init__(self,
                 model_name: str,
                 model_class: Any,
                 eval_key: str,
                 exmp_imgs: Any,
                 lr: float = 1e-3,
                 weight_decay: float = 0.01,
                 seed: int = 42,
                 check_val_every_n_epoch: int = 1,
                 **model_hparams):
        """
        Module for summarizing all common training functionalities.

        Inputs:
            model_name - Folder name in which to save the checkpoints
            model_class - Module class of the model to train
            eval_key - Name of the metric to check for saving the best model
            exmp_imgs - Example imgs, used as input to initialize the model
            lr - Learning rate of the optimizer to use
            weight_decay - Weight decay to use in the optimizer
            seed - Seed to use in the model initialization
            check_val_every_n_epoch - With which frequency to validate the model
        """
        super().__init__()
        self.lr = lr
        self.weight_decay = weight_decay
        self.seed = seed
        self.check_val_every_n_epoch = check_val_every_n_epoch
        # Create empty model. Note: no parameters yet
        self.eval_key = eval_key
        self.model_name = model_name
        self.model = model_class(**model_hparams)
        # Prepare logging
        self.log_dir = os.path.join(CHECKPOINT_PATH, f'{self.model_name}/')
        self.logger = SummaryWriter(log_dir=self.log_dir)
        # Create jitted training and eval functions
        self.create_functions()
        # Initialize model
        self.init_model(exmp_imgs)

    def create_functions(self):
        # To be implemented in sub-classes
        raise NotImplementedError

    def init_model(self, exmp_imgs):
        # Initialize model
        rng = random.PRNGKey(self.seed)
        rng, init_rng = random.split(rng)
        variables = self.model.init(init_rng, exmp_imgs)
        self.state = TrainState(step=0,
                                apply_fn=self.model.apply,
                                params=variables['params'],
                                batch_stats=variables.get('batch_stats'),
                                rng=rng,
                                tx=None, opt_state=None)

    def init_optimizer(self, num_epochs, num_steps_per_epoch):
        # By default, we decrease the learning rate with cosine annealing
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.lr,
            warmup_steps=0.0,
            decay_steps=int(num_epochs * num_steps_per_epoch),
            end_value=2e-2 * self.lr
        )
        optimizer = optax.adamw(lr_schedule, weight_decay=self.weight_decay)
        self.create_train_state(optimizer)

    def create_train_state(self, optimizer):
        # Initialize training state
        self.state = TrainState.create(apply_fn=self.state.apply_fn,
                                       params=self.state.params,
                                       batch_stats=self.state.batch_stats,
                                       rng=self.state.rng,
                                       tx=optimizer)

    def train_model(self, train_loader, val_loader, num_epochs=200):
        # Train model for defined number of epochs
        # We first need to create optimizer and the scheduler for the given number of epochs
        self.init_optimizer(num_epochs, len(train_loader))
        # Track best eval metric
        best_eval = 0.0
        for epoch_idx in tqdm(range(1, num_epochs + 1)):
            self.train_epoch(train_loader, epoch=epoch_idx)
            if epoch_idx % self.check_val_every_n_epoch == 0:
                eval_metrics = self.eval_model(val_loader)
                for key in eval_metrics:
                    self.logger.add_scalar(f'val/{key}', eval_metrics[key], global_step=epoch_idx)
                if eval_metrics[self.eval_key] >= best_eval:
                    best_eval = eval_metrics[self.eval_key]
                    self.save_model(step=epoch_idx)
                self.logger.flush()

    def train_epoch(self, data_loader, epoch):
        # Train model for one epoch, and log avg metrics
        metrics = defaultdict(float)
        for batch in tqdm(data_loader, desc='Training', leave=False):
            self.state, batch_metrics = self.train_step(self.state, batch)
            for key in batch_metrics:
                metrics[key] += batch_metrics[key]
        num_train_steps = len(data_loader)
        for key in metrics:
            avg_val = metrics[key].item() / num_train_steps
            self.logger.add_scalar('train/' + key, avg_val, global_step=epoch)

    def eval_model(self, data_loader):
        # Test model on all images of a data loader and return avg metrics
        metrics = defaultdict(float)
        count = 0
        for batch_idx, batch in enumerate(data_loader):
            batch_metrics = self.eval_step(self.state, random.PRNGKey(batch_idx), batch)
            batch_size = (batch[0] if isinstance(batch, (tuple, list)) else batch).shape[0]
            count += batch_size
            for key in batch_metrics:
                metrics[key] += batch_metrics[key] * batch_size
        metrics = {key: metrics[key].item() / count for key in metrics}
        return metrics

    def save_model(self, step=0):
        # Save current model at certain training iteration
        checkpoints.save_checkpoint(ckpt_dir=self.log_dir,
                                    target={'params': self.state.params,
                                            'batch_stats': self.state.batch_stats},
                                    step=step,
                                    overwrite=True)

    def load_model(self, pretrained=False):
        # Load model. We use different checkpoint for pretrained models
        if not pretrained:
            state_dict = checkpoints.restore_checkpoint(ckpt_dir=self.log_dir, target=None)
        else:
            state_dict = checkpoints.restore_checkpoint(
                ckpt_dir=os.path.join(CHECKPOINT_PATH, f'{self.model_name}.ckpt'), target=None)
        num_params = sum([np.prod(p.shape) for p in jax.tree_leaves(state_dict)])
        self.state = TrainState.create(apply_fn=self.state.apply_fn,
                                       params=state_dict['params'],
                                       batch_stats=state_dict['batch_stats'],
                                       rng=self.state.rng,
                                       tx=self.state.tx if self.state.tx else optax.sgd(self.lr)  # Default optimizer
                                       )

    def checkpoint_exists(self):
        # Check whether a pretrained model exist
        return os.path.isfile(os.path.join(CHECKPOINT_PATH, f'{self.model_name}.ckpt'))