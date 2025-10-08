import jax
from jax import random

from data.augmentations import parallel_augment
from models.simclr import SimCLR
from training.trainer import TrainerModule


class SimCLRTrainer(TrainerModule):

    def __init__(self, **kwargs):
        super().__init__(model_name='SimCLR',
                         model_class=SimCLR,
                         eval_key='acc_top5',
                         **kwargs)

    def create_functions(self):
        # Function to calculate the InfoNCE loss for a batch of images
        def calculate_loss(params, batch_stats, rng, batch, train):
            batch = parallel_augment(rng, batch)
            outs = self.model.apply({'params': params, 'batch_stats': batch_stats},
                                    batch,
                                    train=train,
                                    mutable=['batch_stats'] if train else False)
            (loss, metrics), new_model_state = outs if train else (outs, None)
            return loss, (metrics, new_model_state)

        # Training function
        def train_step(state, batch):
            rng, forward_rng = random.split(state.rng)
            loss_fn = lambda params: calculate_loss(params,
                                                    state.batch_stats,
                                                    forward_rng,
                                                    batch,
                                                    train=True)
            (_, (metrics, new_model_state)), grads = jax.value_and_grad(loss_fn,
                                                                        has_aux=True)(state.params)
            # Update parameters, batch statistics and PRNG key
            state = state.apply_gradients(grads=grads,
                                          batch_stats=new_model_state['batch_stats'],
                                          rng=rng)
            return state, metrics

        # Eval function
        def eval_step(state, rng, batch):
            _, (metrics, _) = calculate_loss(state.params,
                                             state.batch_stats,
                                             rng,
                                             batch,
                                             train=False)
            return metrics

        # jit for efficiency
        self.train_step = jax.jit(train_step)
        self.eval_step = jax.jit(eval_step)