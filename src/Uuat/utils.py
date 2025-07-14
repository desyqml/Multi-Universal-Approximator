import jax.numpy as jnp
import pennylane as qml


def gaussian_kernel(D, sigma):
    return jnp.exp(-D / (2 * sigma))

def cdist(x: jnp.ndarray, y: jnp.ndarray):
  return jnp.sqrt(jnp.sum((x[:, None] - y[None, :]) ** 2, -1))

dev1 = qml.device('default.qubit', wires=1)