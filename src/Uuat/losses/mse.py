import jax.numpy as jnp


def mse(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Mean Squared Error loss."""
    return jnp.mean((x - y) ** 2)
