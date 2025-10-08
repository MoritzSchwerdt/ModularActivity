import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
import flaxmodels
import optax

from copy import deepcopy
from typing import Dict, Any
from typing import Dict, Tuple

class SimCLRWithPosition(nn.Module):
    hidden_dim: int
    temperature: float
    exmp_imgs: Any
    lambda_spatial: float = 0.1
    key: jax.Array = None  # PRNGKey for reproducibility

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
        key = self.key if self.key is not None else random.PRNGKey(0)

        # Derive (C, H, W) from each activation tensor (no training, no gradients)
        exmp_activations = self.convnet(self.exmp_imgs, train=False)
        self.model_units = {
            k: v.shape[1:] for k, v in exmp_activations.items() if isinstance(v, jnp.ndarray)
        }
        self.network_positions = NetworkPositions("ResNet18_Positions")
        self.network_positions.createNetworkPositions(key, self.model_units)

    # ===============================================================
    # == Loss Components ==
    # ===============================================================

    def _simclr_loss(self, feats):
        """Compute InfoNCE contrastive loss."""
        cos_sim = optax.cosine_similarity(feats[:, None, :], feats[None, :, :])
        cos_sim /= self.temperature

        diag_range = jnp.arange(feats.shape[0])
        cos_sim = cos_sim.at[diag_range, diag_range].set(-9e15)

        shifted_diag = jnp.roll(diag_range, feats.shape[0] // 2)
        pos_logits = cos_sim[diag_range, shifted_diag]
        nll = -pos_logits + nn.logsumexp(cos_sim, axis=-1)
        nll = nll.mean()

        metrics = {"loss_simclr": nll}
        return nll, metrics

    def _spatial_loss(self, key, model_feats):
        """Compute cortical spatial correlation loss."""
        spatial_blocks = list(self.network_positions.positions.keys())
        total_spatial_loss = 0.0

        for block in spatial_blocks:
            if block not in model_feats:
                continue

            feats = model_feats[block]  # [B, C, H, W]
            feats = feats.mean(axis=0)  # avg over batch → [C, H, W]
            feats = feats.reshape(feats.shape[0], -1).T  # [N, C]

            positions = self.network_positions.positions[block]
            neighborhoods = self.network_positions.neighborhood_indices[block]

            key, subkey = jax.random.split(key)
            spatial_loss = spatial_correlation_loss(subkey, feats.T, positions, neighborhoods)
            total_spatial_loss += spatial_loss

        return total_spatial_loss

    # ===============================================================
    # == Main forward ==
    # ===============================================================
    def __call__(self, imgs, key, train=True):
        """Compute combined SimCLR + spatial regularization loss."""
        model_feats = self.convnet(imgs, train=train)
        feats = self.head(model_feats["fc"])

        # SimCLR contrastive objective
        simclr_loss, metrics = self._simclr_loss(feats)

        # Spatial regularization term
        spatial_loss = self._spatial_loss(key, model_feats)

        total_loss = simclr_loss + self.lambda_spatial * spatial_loss
        metrics["loss_spatial"] = spatial_loss
        metrics["loss_total"] = total_loss
        return total_loss, metrics

    def encode(self, imgs, train=False):
        model_feats = self.convnet(imgs, train=train)
        return model_feats["block4_1"].mean(axis=(1, 2))
