"""
Utility functions.

This module contains utility functions for computing distances, kernels, and
other functions used in the loss functions and evaluation metrics.
"""

import jax.numpy as jnp
import numpy as np
import pennylane as qml
from scipy.spatial.distance import pdist


def gaussian_kernel(D: jnp.ndarray, sigma: float) -> jnp.ndarray:
    """
    Gaussian kernel function.

    Parameters
    ----------
    D : jnp.ndarray
        Distance matrix.
    sigma : float
        Standard deviation of the kernel.

    Returns
    -------
    jnp.ndarray
        Kernel values.
    """
    return jnp.exp(-D / (2 * sigma))


def cdist(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the pairwise Euclidean distances between two point clouds.

    Parameters
    ----------
    x : jnp.ndarray
        First point cloud.
    y : jnp.ndarray
        Second point cloud.

    Returns
    -------
    jnp.ndarray
        Pairwise distances.
    """
    return jnp.sqrt(jnp.sum((x[:, None] - y[None, :]) ** 2, -1))


def median_dist(x: jnp.ndarray, subset_size: int) -> float:
    """
    Compute the median distance between a subset of points in a point cloud.

    Parameters
    ----------
    x : jnp.ndarray
        Point cloud.
    subset_size : int
        Size of the subset.

    Returns
    -------
    float
        Median distance.
    """
    x_sub = x[np.random.choice(len(x), subset_size, replace=False)]
    pairwise_dist = pdist(x_sub, metric="euclidean")
    return float(jnp.median(pairwise_dist))


dev1 = qml.device("default.qubit", wires=1)

COLORS = {
    "mse": "dodgerblue",
    "corr": "orange",
    "mmd": "violet",
}

