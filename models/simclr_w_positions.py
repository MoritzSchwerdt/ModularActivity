from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
from frozendict import frozendict
from jax import random
from flax import linen as nn
import flaxmodels
import optax

from config import POSITION_KEY, POSITION_PATH#, LAMBDA_SPATIAL
from positions.NetworkPositions import NetworkPositions
from types import MappingProxyType

from copy import deepcopy
from typing import Dict, Any
from typing import Dict, Tuple
from losses.spatial_loss import spatial_correlation_loss, \
    spatial_loss_jit, SpatialData, compute_effective_dimensionality  # , spatial_loss

class SimCLRWithPosition(nn.Module):
    hidden_dim: int
    temperature: float
    init_imgs: Any
    loss_params: Dict[str, Any]

    def setup(self):
        """Setup ResNet, projection head, and cortical positions."""
        self.convnet = flaxmodels.ResNet18(
            output='activations',
            pretrained=False,
            normalize=False,
            num_classes=4 * self.hidden_dim
        )

        self.head = nn.Sequential([
            nn.relu,
            nn.Dense(self.hidden_dim)
        ])

        # Derive (C, H, W) from each activation tensor (no training, no gradients)
        exmp_activations = self.convnet(self.init_imgs, train=False)
        self.model_units = {
            k: jnp.array(v.shape[1:], dtype=jnp.int32)
            for k, v in exmp_activations.items()
            if isinstance(v, jnp.ndarray)
        }

        positions_name = "ResNet18_Positions"
        positions_path = Path(POSITION_PATH) / f"{positions_name}.pkl"

        if positions_path.exists():
            print(f"[INFO] Loading existing NetworkPositions from: {positions_path}")
            network_positions = NetworkPositions.load(positions_path)
        else:
            print(f"[INFO] Creating new NetworkPositions → {positions_name}")

            network_positions = NetworkPositions(positions_name)
            network_positions.createNetworkPositions(random.PRNGKey(POSITION_KEY), self.model_units)
            network_positions.save(positions_path)
            print(f"[INFO] Saved new NetworkPositions to: {positions_path}")

        # Store network positions
        self.network_positions = network_positions

        layers = list(network_positions.positions.keys())
        self.layer_positions = tuple(network_positions.positions[k] for k in layers)
        self.layer_neighborhoods = tuple(network_positions.neighborhood_indices[k] for k in layers)
        self.blocks_radius = tuple(network_positions.neighborhood_widths[k] for k in layers)

        # Store static spatial data container
        self.spatial_data = SpatialData(self.layer_positions, self.layer_neighborhoods, self.blocks_radius)

        print('lol')

    def init_positions(self, params, exmp_imgs, key):
        # This happens after init, so params exist
        activations = self.apply({"params": params}, exmp_imgs, method=lambda m, x: m.convnet(x, train=False))
        model_units = {k: jnp.array(v.shape[1:], dtype=jnp.int32) for k, v in activations.items()}
        net_pos = NetworkPositions("ResNet18_Positions")
        net_pos.createNetworkPositions(key, model_units)
        self.network_positions = net_pos

    # ===============================================================
    # == Loss Components ==
    # ===============================================================

    def _simclr_loss(self, feats):
        """Compute InfoNCE contrastive loss + ranking metrics."""
        # Compute pairwise cosine similarities
        cos_sim = optax.cosine_similarity(feats[:, None, :], feats[None, :, :])
        cos_sim /= self.temperature

        # Mask self-similarities
        diag_range = jnp.arange(feats.shape[0])
        cos_sim = cos_sim.at[diag_range, diag_range].set(-9e15)

        # Identify positive examples (paired images)
        shifted_diag = jnp.roll(diag_range, feats.shape[0] // 2)
        pos_logits = cos_sim[diag_range, shifted_diag]

        # Compute InfoNCE loss
        nll = -pos_logits + nn.logsumexp(cos_sim, axis=-1)
        nll = nll.mean()

        # === Compute additional ranking metrics ===
        # Construct combined similarity matrix: positive + negatives
        comb_sim = jnp.concatenate(
            [pos_logits[:, None], cos_sim[diag_range]], axis=-1
        )

        # Sort similarities (descending), get ranks of positive examples
        sim_ranks = jnp.argsort(-comb_sim, axis=-1).argmin(axis=-1)

        acc_top1 = (sim_ranks == 0).mean()
        acc_top5 = (sim_ranks < 5).mean()
        acc_mean_pos = (sim_ranks + 1).mean()

        # Metrics dict
        metrics = {
            "loss_simclr": nll,
            "acc_top1": acc_top1,
            "acc_top5": acc_top5,
            "acc_mean_pos": acc_mean_pos,
        }

        return nll, metrics

    def _spatial_loss(self, key, model_feats):
        """Compute cortical spatial correlation loss."""
        spatial_blocks = list(self.network_positions.positions.keys())
        total_spatial_loss = []

        for block in spatial_blocks:
            #if block not in model_feats:
            #    continue

            feats = model_feats[block]  # [B, C, H, W]
            #feats = feats.mean(axis=0)  # avg over batch → [C, H, W]
            feats = feats.reshape(feats.shape[0], -1) # [N, C]
            #flat_feats = model_feats['block4_1'].mean(axis=0).flatten()

            #1: sample selected neighborhood
            #2: calculate distances
            #3: calculate correlation
            positions = self.network_positions.positions[block]
            neighborhoods = self.network_positions.neighborhood_indices[block]
            blocks_neighborhood_radius = self.network_positions.neighborhood_widths[block]
            spatial_loss = spatial_correlation_loss(key, feats, positions, neighborhoods, blocks_neighborhood_radius)

            print(spatial_loss)
            key, subkey = jax.random.split(key)
            spatial_loss = spatial_correlation_loss(subkey, feats, positions, neighborhoods)
            total_spatial_loss += spatial_loss

        return total_spatial_loss



    # ===============================================================
    # == Main forward ==
    # ===============================================================
    #@partial(jax.jit, static_argnames=("train", "summed"))
    def __call__(self, imgs, key,lambda_spatial=0, train=True, summed=True):
        """Compute combined SimCLR + spatial regularization loss."""
        layers = list(self.network_positions.positions.keys())
        conv_outputs = self.convnet(imgs, train=train)
        model_feats_list = tuple(conv_outputs[k] for k in layers)
        #model_neighborhood_feats_list = tuple(conv_outputs[k][self.spatial_data.neighborhoods[k]] for i,k in enumerate(layers))
        #model_neighborhoods = tuple(self.spatial_data.neighborhoodsk[k] for k in layers)
        #[(i,conv_outputs[k].shape) for i,k in enumerate(layers)]

        feats = self.head(conv_outputs["fc"])

        computed_simclr_loss, metrics = self._simclr_loss(feats)

        computed_spatial_loss = spatial_correlation_loss(self.spatial_data, model_feats_list, key, self.loss_params['correlation_mode'])

        # Compute spatial regularization
        #computed_spatial_loss = spatial_loss_jit(self.spatial_data, model_feats_list, key)

        if summed:
            total_loss = computed_simclr_loss + lambda_spatial * computed_spatial_loss
        else:
            total_loss = jnp.stack([computed_simclr_loss, computed_spatial_loss])
        #metrics['total_loss'] = computed_simclr_loss + computed_spatial_loss
        metrics["loss_spatial"] = computed_spatial_loss
        metrics["loss_simclr"] = computed_simclr_loss
        #metrics["loss_total"] = total_loss
        #jax.debug.breakpoint()
        return total_loss, metrics

    def encode(self, imgs, train=False):
        model_feats = self.convnet(imgs, train=train)
        return model_feats["block4_1"].mean(axis=(1, 2))

    def apply_convnet(self, imgs, train: bool = False):
        # call the convnet submodule and return its activations
        return self.convnet(imgs, train=train)

    def get_positions(self):
        return self.network_positions

    def get_spatial_data(self):
        return self.spatial_data

    def save_positions(self, path):
        self.network_positions.save(path)

    def load_positions(self, path):
        self.network_positions = self.network_positions.load(path)