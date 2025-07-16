import jax.numpy as jnp


def corr(x: jnp.ndarray, y: jnp.ndarray) -> float:
    """Computes the correlation loss between two datasets.

    Parameters
    ----------
    set1 : NDArray[jnp.float64]
        First dataset.
    set2 : NDArray[jnp.float64]
        Second dataset.

    Returns
    -------
    float
        Mean squared difference between correlation matrices.
    """
    return float(
        jnp.mean((jnp.corrcoef(x, rowvar=False) - jnp.corrcoef(y, rowvar=False)) ** 2)
    )
