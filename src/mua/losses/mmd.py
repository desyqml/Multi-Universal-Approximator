from typing import List

import jax.numpy as jnp

from ..utils import cdist, gaussian_kernel


def mmd(x: jnp.ndarray, y: jnp.ndarray, p_sigma: List[float]) -> jnp.ndarray:
    """
    Compute the Maximum Mean Discrepancy (MMD) between two sets of samples.

    Parameters
    ----------
    x : jnp.ndarray
        First set of samples.
    y : jnp.ndarray
        Second set of samples.
    p_sigma : List[float]
        List of bandwidths for the Gaussian kernel.

    Returns
    -------
    jnp.ndarray
        The MMD value.
    """
    # Compute squared pairwise distances for both sets
    XX = cdist(x, x + 1e-6) ** 2  # Add small epsilon to avoid zero distance
    YY = cdist(y, y) ** 2
    XY = cdist(x, y) ** 2

    # Compute the sum of Gaussian kernels over multiple bandwidths
    kernels = jnp.array([
        gaussian_kernel(XX, sigma).mean() +
        gaussian_kernel(YY, sigma).mean() -
        2 * gaussian_kernel(XY, sigma).mean()
        for sigma in p_sigma
    ])

    # Average the MMD values across all bandwidths
    return jnp.mean(kernels)

    # Average the MMD values across all bandwidths
    return jnp.mean(kernels)
