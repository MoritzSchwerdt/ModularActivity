from typing import Tuple, List, Dict
from pathlib import Path
from config import *

import numpy as np


import pickle
import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple, Literal


class NetworkPositions:
    # Name of Network
    name: str
    # Dims of Units inside of CNN Network
    dims: Dict[str, Tuple[int]]

    # cordinates mapping blocks to Nx2 matrix x&y coordinates of Units
    centers: Dict[str, jnp.array]
    positions: Dict[str, jnp.array]
    anchors: Dict[str, jnp.array]

    # neighborhood_indices: PxQ matrix mapping Neigborhoods Position of Block onto Cortical Sheet
    neighborhood_indices: Dict[str, jnp.array]

    # width in mm of the neighborhoods we keep this static, but may vary for positions
    neighborhood_widths: Dict[str, jnp.array]


    def __init__(self, name):
        self.name = name

    def createNetworkPositions(self, key, model_units):
        # Creates Network Positions based on CNN Units per Layer
        common_blocks = model_units.keys() & brain_mapping.keys()  # set intersection
        self.dims = {k: model_units[k].shape[1:] for k in common_blocks}
        centers, positions, neighborhood_indices, neighborhood_widths = {}, {}, {}, {}
        anchors = {}

        for block in common_blocks:
            brain_area = brain_mapping[block]
            tissue_size = TISSUE_SIZES[brain_area]
            neighborhood_width = NEIGHBORHOOD_WIDTHS[brain_area]

            anchors_block, positions_block, rf_radius_block = place_conv_vec(key=key, dims=self.dims[block],
                                                                             pos_lims=(0, tissue_size),
                                                                             offset_pattern='random', rf_overlap=RF_OVERLAP)
            anchors[block] = anchors_block

            positions[block] = positions_block

            centers[block] = sample_centers(key, positions_block, NEIGHBORHOOD_WIDTHS[brain_area] / 2, N_NEIGHBORHOODS)

            # possibly adapt radii per center
            radii = jnp.ones(len(centers[block], )) * (neighborhood_width / 2)
            neighborhood_widths[block] = radii

            # positions[block] = getCoordinates(block, tissue_size, neighborhood_width)
            neighborhood_indices[block] = sample_neighbors_batch_vectorized(key, positions_block, centers[block], radii,
                                                                            N_NEIGHBORS)

        self.anchors = anchors
        self.centers = centers
        self.positions = positions
        self.neighborhood_indices = neighborhood_indices
        self.neighborhood_widths = neighborhood_widths

    def save(self, save_dir: Path):
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        path = save_dir / f"{self.name}.pkl"
        with path.open("wb") as stream:
            pickle.dump(self, stream)

    def save_np(self, save_dir: Path):
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True, parents=True)
        path = save_dir / f"{self.name}.npz"
        np.savez(path, **vars(self))

    @classmethod
    def load(cls, path: Path) -> "NetworkPositions":
        path = Path(path)
        assert path.exists(), path

        if path.suffix == ".pkl":
            with path.open("rb") as stream:
                return pickle.load(stream)

        # TODO: fix numpy loading of Positions
        elif path.suffix == ".npz":
            state = np.load(path)

            def _scalar(x):
                return x[()]

            return cls(
                name=_scalar(state["name"]),
                dims=state["dims"],
                centers= state["centers"],
                positions= state["positions"],
                anchors= state["anchors"],
                neighborhood_indices=state["neighborhood_indices"],
                neighborhood_width=_scalar(state["neighborhood_width"]),
            )
        raise ValueError("suffix must be npz or pkl")


def jitter_positions(key, pos: jnp.ndarray, jitter: float = 0.0):
    """Add jitter to positions (JAX)."""
    if jitter == 0.0:
        return pos

    # Add Gaussian noise
    noise = random.normal(key, shape=pos.shape) * jitter
    jittered = pos + noise

    # Squish: normalize range, then rescale to original range
    old_range = jnp.ptp(pos, axis=0)  # (2,)
    new_range = jnp.ptp(jittered, axis=0)
    ratio = new_range / old_range
    jittered_squished = jittered / ratio

    # Slide: align minimums with original
    result = jittered_squished - jnp.mean(jittered_squished, axis=0) + jnp.mean(pos, axis=0)

    # result = jittered_squished - jnp.min(jittered_squished, axis=0) + jnp.min(pos, axis=0)
    return result


def place_conv_vec(
        key: jax.Array,
        dims: Tuple[int, int, int],
        pos_lims: Tuple[float, ...],
        offset_pattern: Literal["random", "grid"] = "random",
        rf_overlap: float = 0.0,
        flatten: bool = True
):
    """
    Places units in a conv layer on a cortical sheet (square kernels).

    Args:
        key: PRNGKey
        dims: (num_channels, num_x, num_y) feature dimensions
        pos_lims: If (2,), symmetric min/max for both x and y.
                  If (4,), treated as [x0, x1, y0, y1].
        offset_pattern: 'random' or 'grid'
        rf_overlap: fraction between 0 and 1 for RF overlap
        flatten: if True, returns (num_channels*num_x*num_y, 2),
                 else (num_channels, num_x, num_y, 2)
        return_rf_radius: if True, also returns the RF radius

    Returns:
        positions: array of shape (..., 2) with unit positions
        rf_radius (optional): scalar radius of receptive fields
    """

    num_chan, num_x, num_y = dims

    def compute_rf_centers(num_rfs, rf_overlap, lims):
        """Helper for 1D axis (JAX)."""
        map_width = jnp.ptp(jnp.array(lims))
        rf_width = map_width / (num_rfs - (num_rfs * rf_overlap) + rf_overlap)
        rf_radius = rf_width / 2.0
        centers = jnp.linspace(lims[0] + rf_radius, lims[1] - rf_radius, num_rfs)
        return centers, rf_radius

    if len(pos_lims) == 2:
        x_centers, x_radius = compute_rf_centers(num_x, rf_overlap, pos_lims)
        y_centers, y_radius = compute_rf_centers(num_y, rf_overlap, pos_lims)
    elif len(pos_lims) == 4:
        x_centers, x_radius = compute_rf_centers(num_x, rf_overlap, pos_lims[:2])
        y_centers, y_radius = compute_rf_centers(num_y, rf_overlap, pos_lims[2:])
    else:
        raise ValueError("pos_lims must be length 2 or 4")

    rf_radius = jnp.maximum(x_radius, y_radius)  # square kernels
    xx, yy = jnp.meshgrid(x_centers, y_centers, indexing="xy")
    anchors = jnp.stack((xx.ravel(), yy.ravel()), axis=1)  # (num_x*num_y, 2)

    # Apply channel-specific offsets
    if offset_pattern == "random":
        key, subkey = random.split(key)
        offsets = random.uniform(
            subkey,
            shape=(anchors.shape[0], num_chan, 2),
            minval=-rf_radius,
            maxval=rf_radius,
        )
    elif offset_pattern == "grid":
        grid_size = int(jnp.ceil(jnp.sqrt(num_chan)))
        gx = jnp.linspace(-rf_radius, rf_radius, grid_size)
        gy = jnp.linspace(-rf_radius, rf_radius, grid_size)
        gxx, gyy = jnp.meshgrid(gx, gy, indexing="xy")
        grid_offsets = jnp.stack((gxx.ravel(), gyy.ravel()), axis=1)[:num_chan]
        offsets = jnp.tile(grid_offsets[None, :, :], (anchors.shape[0], 1, 1))
    else:
        raise ValueError(f"Offset pattern '{offset_pattern}' not recognized")

    # Combine anchors + offsets
    anchors_expanded = anchors[:, None, :]  # (num_x*num_y, 1, 2)
    positions = anchors_expanded + offsets  # (num_x*num_y, num_chan, 2)

    # Move channels first
    positions = jnp.swapaxes(positions, 0, 1)  # (num_chan, num_x*num_y, 2)
    positions = jitter_positions(key, positions, 0.3)

    # Reshape to final form
    if flatten:
        positions = positions.reshape((-1, 2))  # (num_chan*num_x*num_y, 2)
    else:
        positions = positions.reshape((num_chan, num_x, num_y, 2))

    return anchors, positions, rf_radius


def sample_centers(key, positions, avg_radius, n_neighborhoods):
    key_x, key_y = jax.random.split(key)
    centers_xs = jax.random.uniform(key=key_x, minval=jnp.min(positions[:, 0]) + avg_radius,
                                    maxval=jnp.max(positions[:, 0]) - avg_radius, shape=(n_neighborhoods,))
    centers_ys = jax.random.uniform(key=key_y, minval=jnp.min(positions[:, 1]) + avg_radius,
                                    maxval=jnp.max(positions[:, 1]) - avg_radius, shape=(n_neighborhoods,))

    return jnp.stack((centers_xs, centers_ys), axis=1)


def sample_neighbors_batch_vectorized(key, positions, centers, radii, n_neighbors):
    """
    consider each center as a neighborhood and computes all Positions within radius for the center
    Samples iid (possibly related to distance) neighbors
    If amount neighbors < required amount, resample otherwise no resampling

    Inputs:
        positions: N x 2 position matrix
        centers: n_neighborhoods Amount of Centers that compute neighborhoods
        radii: radius for a neighborhood (width will be 2 * radius) length of centers
        n_neighbors: how many neighbors are part of the Neighborhood (specific center)
    """
    num_centers = centers.shape[0]

    # Compute squared distances: (num_centers, num_positions)
    # sq_distances = jnp.sum((centers[:, None] - positions[None, :]) ** 2, axis=-1)

    sq_distances = jnp.linalg.norm((centers[:, None] - positions[None, :]), ord=2, axis=-1)

    # Mask of valid neighbors
    mask = sq_distances <= radii[:, None]
    n_valid = jnp.sum(mask, axis=1)

    # potential probs based on distance to center:
    future_probs = mask * sq_distances

    # Split keys for each center
    keys = random.split(key, num_centers)

    def sample_center(key, mask_row, n_valid_row):
        key, subkey = random.split(key)
        probs = mask_row / n_valid_row

        def sample_no_replace(_):
            return random.choice(subkey, a=mask_row.shape[0], shape=(n_neighbors,), replace=False, p=probs)

        def sample_replace(_):
            return random.choice(subkey, a=mask_row.shape[0], shape=(n_neighbors,), replace=True, p=probs)

        return jax.lax.cond(n_valid_row >= n_neighbors, sample_no_replace, sample_replace, operand=None)

    # vmap over centers
    sampled_indices = jax.vmap(sample_center, in_axes=(0, 0, 0))(keys, mask, n_valid)
    return sampled_indices



