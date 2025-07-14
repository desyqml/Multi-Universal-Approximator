from typing import List

import jax.numpy as jnp

from ..utils import cdist, gaussian_kernel


def mmd(x: jnp.ndarray, y: jnp.ndarray, p_sigma: List[float]) -> jnp.ndarray:
    # Compute squared pairwise distances
    XX = cdist(x, x + 1e-6) ** 2
    YY = cdist(y, y) ** 2
    XY = cdist(x, y) ** 2

    # Compute the sum of Gaussian kernels over multiple bandwidths

    kernels = jnp.array([
        gaussian_kernel(XX, sigma).mean() +
        gaussian_kernel(YY, sigma).mean() -
        2 * gaussian_kernel(XY, sigma).mean()
        for sigma in p_sigma
    ])

    # Average across all bandwidths
    return jnp.mean(kernels)