import jax.numpy as jnp
import numpy as np
import optax
import tqdm
from jax import jit, value_and_grad, vmap

from ..losses.mse import mse
from ..unitaries.Uuat import Uuat

x_regression = jnp.linspace(-1, 1, 100)

jv_Uuat = jit(vmap(Uuat, in_axes=(0, None, None, None)))

def train(target_fun, depth, num_epochs, lr = 0.1):
    def loss_fn(params, x, y_true):
        p_omega, p_alpha, p_phi = params
        y_pred = jv_Uuat(x, p_omega, p_alpha, p_phi)
        return mse(y_true, y_pred)

    @jit
    def train_step(params, opt_state, x, y_true):
        loss, grads = value_and_grad(loss_fn)(params, x, y_true)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    target_vals = target_fun(x_regression)
    assert (target_vals >= -1).all() and (target_vals <= 1).all()

    # Storage for parameters and losses during training for animation
    params_history = []
    loss_history = []
    # Convert parameters to jnp arrays for JAX compatibility
    p_omega = jnp.array(np.random.rand(depth))
    p_alpha = jnp.array(np.random.rand(depth))
    p_phi = jnp.array(np.random.rand(depth))

    # Initialize optimizer (Adam with learning rate)
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init((p_omega, p_alpha, p_phi))
    progress = tqdm.tqdm(range(num_epochs))
    
    params = (p_omega, p_alpha, p_phi)
    for epoch in progress:
        params, opt_state, loss = train_step(params, opt_state, x_regression, target_vals)
        params_history.append(params)
        loss_history.append(loss)
        progress.set_description(f"MSE: {loss}")

    return params, params_history, loss_history
