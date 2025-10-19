from functools import partial

import jax
import optax
from jax import random
import jax.numpy as jnp

from data.augmentations import parallel_augment
from models.simclr import SimCLR
from models.simclr_w_positions import SimCLRWithPosition
from training.trainer import TrainerModule


class SimCLRPositionTrainer(TrainerModule):

    def __init__(self, **kwargs):
        super().__init__(
            model_class=SimCLRWithPosition,
            eval_key='acc_top5',
            **kwargs)

    def create_functions(self):
        # Function to calculate the InfoNCE loss for a batch of images
        def calculate_loss(params, batch_stats, rng, batch, lambda_spatial, train, summed=True):
            batch = parallel_augment(rng, batch)
            outs = self.model.apply({'params': params, 'batch_stats': batch_stats},
                                    batch,
                                    train=train,
                                    summed=summed,
                                    key=rng,
                                    lambda_spatial=lambda_spatial,
                                    mutable=['batch_stats'] if train else False)
            (loss, metrics), new_model_state = outs if train else (outs, None)
            return loss, (metrics, new_model_state)

        # Training function
        @partial(jax.jit, static_argnames=['aggr_mode', 'eps'])
        def train_step(state, batch, lambda_spatial, p_t=None, aggr_mode="scalar", eps=1e-8):
            rng, forward_rng = random.split(state.rng)
            if aggr_mode == "scalar":
                loss_fn = lambda params: calculate_loss(params,
                                                        state.batch_stats,
                                                        forward_rng,
                                                        batch,
                                                        lambda_spatial,
                                                        train=True,
                                                        summed=True)
                (_, (metrics, new_model_state)), grads = jax.value_and_grad(loss_fn,
                                                                            has_aux=True)(state.params)
                # Update parameters, batch statistics and PRNG key
                state = state.apply_gradients(grads=grads,
                                              batch_stats=new_model_state['batch_stats'],
                                              rng=rng)
                return state, metrics

            elif aggr_mode == "scaled_gradients":

                assert p_t is not None, "Must provide p_t for mode='scaled_gradients'."

                loss_fn = lambda params: calculate_loss(params,
                                                        state.batch_stats,
                                                        forward_rng,
                                                        batch,
                                                        jnp.ones(1),
                                                        train=True,
                                                        summed=False)
                # (stacked_loss, (metrics, new_model_state)) = loss_fn(state.params)

                # Evaluate VJP at params
                # stacked loss is stacked [simclr_loss, topo_loss]
                # Evaluate VJP at params (stacked_loss is e.g. [simclr_loss, topo_loss])
                (stacked_loss, (metrics, updated_model_state)), vjp_fun = jax.vjp(loss_fn, state.params)
                unscaled_simclr_loss, unscaled_topo_loss = stacked_loss

                # Build zero-valued aux that matches the exact PyTree structure of (metrics, updated_model_state).
                # We use zeros_like for any array-like leaves; for non-array leaves fall back to 0.
                zero_metrics = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x) if isinstance(x, jnp.ndarray) else 0,
                                                      metrics)
                zero_model_state = jax.tree_util.tree_map(
                    lambda x: jnp.zeros_like(x) if isinstance(x, jnp.ndarray) else 0, updated_model_state)
                aux_zero = (zero_metrics, zero_model_state)

                # Basis vectors for selecting each loss in the stacked loss
                basis = [jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])]

                # Obtain gradients for each loss by calling vjp_fun with matching PyTree structure
                grads_task, grads_topo = [vjp_fun((b, aux_zero))[0] for b in basis]

                # --- Compute norms and scaling factor ---
                norm_task = optax.global_norm(grads_task)
                norm_topo = optax.global_norm(grads_topo)
                k = ((1.0 - p_t) / p_t) * (norm_task / (norm_topo + eps))

                # --- Combine gradients ---
                grads_combined = jax.tree_util.tree_map(
                    lambda g_t, g_s: g_t + k * g_s,
                    grads_task,
                    grads_topo,
                )

                # --- Apply optimizer update ---
                new_state = state.apply_gradients(
                    grads=grads_combined,
                    batch_stats=updated_model_state["batch_stats"],
                    rng=rng
                )

                # --- Log diagnostics ---
                metrics.update({
                    "grad_norm_task": norm_task,
                    "grad_norm_topo": norm_topo,
                    "grad_scale_k": k,
                    "p_target": p_t,
                })

                return new_state, metrics
            else:
                raise ValueError(f"Unknown mode: {aggr_mode}")

        # Eval function
        def eval_step(state, rng, batch):
            _, (metrics, _) = calculate_loss(state.params,
                                             state.batch_stats,
                                             rng,
                                             batch,
                                             jnp.ones(1),
                                             summed=False,
                                             train=False)
            return metrics

        # jit for efficiency
        self.train_step = train_step
        # self.train_step = jax.jit(train_step)
        self.eval_step = jax.jit(eval_step)