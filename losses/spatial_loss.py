from functools import partial
from typing import NamedTuple, Tuple

import jax.numpy as jnp
import jax.random as random
from flax import linen as nn
import jax
import optax

from config import SAMPLED_NEIGHBORHOODS


# ===============================================================
# == PyTree Static Container
# ===============================================================
class SpatialData:
    """Static container for positions and neighborhoods (registered PyTree)."""
    def __init__(self, positions, neighborhoods, radii):
        self.positions = positions
        self.neighborhoods = neighborhoods
        self.radii = radii

    def tree_flatten(self):
        # no dynamic leaves (these are all static)
        return (), (self.positions, self.neighborhoods, self.radii)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        positions, neighborhoods, radii = aux_data
        return cls(positions, neighborhoods, radii)


# Register as pytree
jax.tree_util.register_pytree_node(
    SpatialData,
    lambda obj: ((), (obj.positions, obj.neighborhoods, obj.radii)),  # flatten
    lambda aux_data, children: SpatialData(*aux_data)      # unflatten
)

# Vectorized corrcoef across batch
@jax.jit
def batch_corrcoef(X):
    def corr(x):
        x = x - jnp.mean(x, axis=0)
        cov = (x.T @ x) / (x.shape[0] - 1)
        d = jnp.sqrt(jnp.diag(cov))
        return cov / jnp.outer(d, d)
    return jax.vmap(corr)(X)

# ===============================================================
# == Loss Components ==
# ===============================================================
"""
def spatial_correlation_loss(
    key: jax.Array, features: jnp.ndarray, positions: jnp.ndarray, neighborhoods: jnp.ndarray
) -> float:
    
    Spatial correlation loss in JAX.

    L = 0.5 * (1 - corr(response_similarity, spatial_similarity))

    Args:
        key: PRNG key
        features: [batch, N] or similar
        positions: [N, 2] array of spatial coords
        neighborhoods: [M, K] array of indices into positions
    

    # pick a random neighborhood
    idx = random.randint(key, shape=(20,), minval=0, maxval=neighborhoods.shape[0])
    indices = neighborhoods[idx,:]

    # flatten CHW → [batch, N] and select neighborhood
    neighborhood_features = features[:, indices]
    neighborhood_positions = positions[indices]

    # pairwise Euclidean distances
    diffs = positions[:, None, :] - positions[None, :, :]
    dists = jnp.linalg.norm(diffs, axis=-1)

    # spatial similarity kernel
    distance_similarity = 1.0 / (1.0 + dists)

    # response similarity (corrcoef is in JAX already!)
    response_similarity = batch_corrcoef(neighborhood_features)
    response_similarity = jnp.corrcoef(neighborhood_features.T)

    # extract lower triangle (k=-1)
    tri_idx = jnp.tril_indices(response_similarity.shape[0], k=-1)
    r = response_similarity[tri_idx]
    D = distance_similarity[tri_idx]

    # correlation between flattened values
    sim_align = jnp.corrcoef(jnp.stack([r, D]))[0, 1]

    # final loss
    return 0.5 * (1.0 - sim_align)
"""

@jax.jit
def spatial_correlation_loss_jit(
    key: jax.Array,
    features: jnp.ndarray,         # [N] for single batch
    positions: jnp.ndarray,        # [N, 2]
    neighborhoods: jnp.ndarray     # [M, K]
) -> float:
    """
    Spatial correlation loss for a single batch.
    """
    # pick a random neighborhood
    neigh_idx = random.randint(key, shape=(100), minval=0, maxval=neighborhoods.shape[0])
    indices = neighborhoods[neigh_idx]  # [K]

    # neighborhood features & positions
    neighborhood_features = features[indices]  # [K]
    neighborhood_positions = positions[indices]  # [K, 2]

    # response similarity (correlation matrix)
    response_similarity = jnp.corrcoef(neighborhood_features)  # [K, K]

    # spatial similarity (pairwise distances)
    diffs = neighborhood_positions[:, None, :] - neighborhood_positions[None, :, :]
    dists = jnp.linalg.norm(diffs, axis=-1)
    distance_similarity = 1.0 / (1.0 + dists)  # [K, K]

    # lower triangle indices
    tri_idx = jnp.tril_indices(distance_similarity.shape[0], k=-1)
    r = response_similarity[tri_idx]
    D = distance_similarity[tri_idx]

    # alignment correlation
    sim_align = jnp.corrcoef(jnp.stack([r, D]))[0, 1]

    return 0.5 * (1.0 - sim_align)


@partial(jax.jit, static_argnames=("spatial_data","mode","r0",))
def spatial_loss_jit(spatial_data: SpatialData, model_feats, key, mode="mexican_hat", r0=0.4):
    """
    Compute total spatial correlation loss across all layers and batches.
    spatial_data: static SpatialData (positions, neighborhoods)
    model_feats: tuple of feature maps (dynamic)
    """
    total_losses = []
    debug_info = []

    #Iterate CNN Unit Blocks
    for i, feats in enumerate(model_feats):
        feats = feats.reshape(feats.shape[0], -1)  # [B, N]
        position = spatial_data.positions[i]
        neighborhood = spatial_data.neighborhoods[i]

        keys = random.split(key, feats.shape[0])
        key, _ = random.split(key)

        neigh_idx = random.randint(key, shape=(SAMPLED_NEIGHBORHOODS), minval=0, maxval=neighborhood.shape[0])
        indices = neighborhood[neigh_idx]

        neighborhood_features = feats[indices]  # [K]
        neighborhood_positions = position[indices]  # [K, 2]

        # Compute L2 Dist Matrix
        diffs = position[:, None, :] - position[None, :, :]
        dists = jnp.linalg.norm(diffs, axis=-1)

        neighborhood_dists =  dists[indices[:, :, None], indices[:, None, :]]

        eps = 1e-8
        #Batched Corrcoeff
        selected_reponses = jnp.transpose(feats[:,indices], (1,0,2))
        response_similarity = jax.vmap(lambda x: jnp.corrcoef(x + eps, rowvar=False))(selected_reponses)

        def neighborhood_loss(rmat, dmat, eps=1e-8):
            tri_idx = jnp.tril_indices(rmat.shape[0], k=-1)
            r = rmat[tri_idx]
            D = dmat[tri_idx]
            return jnp.corrcoef(jnp.stack([r, D]) + eps)[0, 1]

        similarity_alignment = jax.vmap(neighborhood_loss)(response_similarity, neighborhood_dists)  # [SAMPLED_NEIGHBORHOODS]

        batch_losses = jnp.mean(0.5 * (1.0 - similarity_alignment))

        #batched_corr = jax.vmap(lambda feat_batch: jax.vmap(jnp.corrcoef)(feat_batch))  feats[:,indices].shape

        #jax.vmap(lambda x: jnp.corrcoef(x, rowvar=False))(selected)
        #feats[indices]
        # jnp.transpose(feats[:,indices], (1,2,0)).shape

        #Compute Batched Corrcoeff
        #jax.vmap(lambda x: jnp.corrcoef(x, rowvar=False))(jnp.transpose(feats[:,indices], (1,0,2)))

        #dists[indices[:,:,None],indices[:,None,:]] Get Batched Distance Matrix
        """batch_losses = jax.vmap(
            spatial_correlation_loss_jit,
            in_axes=(0, 0, None, None)
        )(keys, feats, position, neighborhood)"""

        #jax.debug.breakpoint()
        total_losses.append(batch_losses)

        # Store debug information
        debug_info.append({
            "layer": i,
            "feats_shape": feats.shape,
            "position_shape": position.shape,
            "neighborhood_shape": neighborhood.shape,
            "indices_shape": indices.shape,
            "neigh_idx": neigh_idx,
            "neighborhood_dists_min": jnp.min(neighborhood_dists),
            "neighborhood_dists_max": jnp.max(neighborhood_dists),
            "response_similarity_min": jnp.min(response_similarity),
            "response_similarity_max": jnp.max(response_similarity),
            "similarity_alignment": similarity_alignment,
            "batch_loss": batch_losses
        })

    #jax.debug.breakpoint()
    return jnp.mean(jnp.array(total_losses))

@partial(jax.jit, static_argnames=("spatial_data","mode","r0",))
def spatial_correlation_loss(spatial_data : SpatialData,model_feats,key, mode="TDANN", r0=0.4, eps=1e-8):
    """
    Fully JIT-compatible spatial correlation loss with multiple target correlation modes.
    Modes: ['positive', 'mexican_hat', 'perfect_corr', 'perfect_anti_corr']
    """

    def get_target_correlation(distances, mode, r0, radius):
        #jax.debug.breakpoint()
        rel_distances = distances / (radius[:,None,None] + eps)

        positive = jnp.exp(-4 * rel_distances)
        #mexican = jnp.exp(-(rel_distances / r0) ** 2) - 0.6 * jnp.exp(-(rel_distances / (2 * r0)) ** 2)
        sigma = 0.35
        mexican = (1 - (rel_distances**2 / sigma**2)) * jnp.exp(-rel_distances**2 / (2 * sigma**2))

        #tdann = 1.0 / (distances + 1.0)
        tdann = 1 - rel_distances
        tdann_corr = rel_distances  # optional variant if you want raw correlation

        target = jnp.select(
            [
                mode == "positive",
                mode == "mexican_hat",
                mode == "TDANN",
                mode == "TDANN_corr",
            ],
            [positive, mexican, tdann, tdann_corr],
            default=positive,
        )
        return target

    def neighborhood_loss(response_corr, neighborhood_dist, target_corr):
        tri_idx = jnp.tril_indices(response_corr.shape[0], k=-1)
        #add epsilon for numeric stability, as 0 variance will break into nan
        r = response_corr[tri_idx]
        r = r + eps * (1.0 + jnp.arange(r.shape[0], dtype=r.dtype))
        t = target_corr[tri_idx]
        t = t + eps * (1.0 + jnp.arange(t.shape[0], dtype=t.dtype))

        #corrcoeff computes correlation matrix, looking at covariance
        corr_alignment = jnp.corrcoef(jnp.stack([r, t]))[0, 1]
        loss = (1.0 - corr_alignment) / 2.0
        return loss, corr_alignment

    total_losses = []
    debug_info = []

    for i, feats in enumerate(model_feats):
        feats = feats.reshape(feats.shape[0], -1)
        position = spatial_data.positions[i]
        neighborhood = spatial_data.neighborhoods[i]
        radius = spatial_data.radii[i]

        key, subkey = random.split(key)
        neigh_idx = random.randint(subkey, shape=(SAMPLED_NEIGHBORHOODS,), minval=0, maxval=neighborhood.shape[0])
        indices = neighborhood[neigh_idx]
        sampled_radii = radius[neigh_idx]

        diffs = position[:, None, :] - position[None, :, :]
        dists = jnp.linalg.norm(diffs, axis=-1)
        neighborhood_dists = dists[indices[:, :, None], indices[:, None, :]]

        selected_responses = jnp.transpose(feats[:, indices], (1, 0, 2))
        #response_similarity = jax.vmap(lambda x: jnp.corrcoef(x + eps, rowvar=False))(selected_responses)

        response_similarity = jax.vmap(
            lambda x, k: jnp.corrcoef(x + eps * random.normal(k, x.shape), rowvar=False)
        )(selected_responses, random.split(subkey, selected_responses.shape[0]))

        jax.debug.breakpoint()

        target_corr = get_target_correlation(neighborhood_dists, mode, r0, sampled_radii)
        losses, corr_alignment = jax.vmap(neighborhood_loss, in_axes=(0,0,0))(response_similarity, neighborhood_dists, target_corr)
        total_losses.append(jnp.mean(losses))
        # store scalar debug info

        """debug_info.append({
            "layer": i,
            "feats_shape": feats.shape,
            "position_shape": position.shape,
            "neighborhood_shape": neighborhood.shape,
            "indices_shape": indices.shape,
            "neigh_idx": neigh_idx,
            "neighborhood_dists_min": jnp.min(neighborhood_dists),
            "neighborhood_dists_max": jnp.max(neighborhood_dists),
            "response_similarity_min": jnp.min(response_similarity),
            "response_similarity_max": jnp.max(response_similarity),
            "corr_alignment": corr_alignment,
            "layer_loss": jnp.mean(losses),
        })"""

    #jax.debug.breakpoint()
    return jnp.mean(jnp.stack(total_losses))

@jax.jit
def effective_dimensionality(features, eps=1e-8):
    """
    Computes effective dimensionality (participation ratio)
    for feature matrices with n_features >> n_samples.
    Uses SVD for numerical stability.
    """
    # Center features
    features = features - jnp.mean(features, axis=0, keepdims=True)
    features = features.reshape(features.shape[0], -1)
    features = jnp.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    features = features.astype(jnp.float32)

    n_samples = features.shape[0]

    # Compute singular values (economy SVD)
    s = jnp.linalg.svd(features, compute_uv=False)

    # Convert to covariance eigenvalues
    eigvals = (s ** 2) / jnp.maximum(n_samples - 1, 1)
    eigvals = jnp.maximum(eigvals, eps)

    # Participation ratio
    numerator = jnp.sum(eigvals) ** 2
    denominator = jnp.sum(eigvals ** 2)
    D_eff = numerator / denominator

    return D_eff.astype(jnp.float32)


@jax.jit
def compute_effective_dimensionality(features_list):
    """Compute mean effective dimensionality across layers."""
    eff_dims = [effective_dimensionality(f) for f in features_list]
    return jnp.mean(jnp.stack(eff_dims)).astype(jnp.float32)