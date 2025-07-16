import jax.numpy as jnp


def mse(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the Mean Squared Error between two arrays of values.

    Parameters
    ----------
    x : jnp.ndarray
        The first array of values.
    y : jnp.ndarray
        The second array of values.

    Returns
    -------
    jnp.ndarray
        The mean squared error between the two arrays.
    """
    return jnp.mean((x - y) ** 2)
