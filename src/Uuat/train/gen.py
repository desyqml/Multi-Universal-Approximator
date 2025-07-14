import jax.numpy as jnp
import numpy as np
import optax
import tqdm
from jax import jit, value_and_grad, vmap

from ..losses.mmd import mmd
from ..unitaries.multiUuat import multi_Uuat

x_regression = jnp.linspace(-1, 1, 100)

jv_multi_Uuat = jit(vmap(multi_Uuat, in_axes=(0, None, None, None)))

def train(pp_Y, depth, num_imgs, num_epochs, lr = 0.1, p_sigma = [0.1, 0.25, 0.5, 0.75, 1], generate = False):
    def loss_fn(params, x, y_target, p_sigma):
        pp_phi, ppp_omega, pp_alpha = params
        y_pred = jv_multi_Uuat(x, pp_phi, ppp_omega, pp_alpha)
        loss_value = mmd(y_pred, y_target, p_sigma)
        return loss_value

    @jit
    def train_step(params, opt_state, x, y_target, p_sigma):
        loss, grads = value_and_grad(loss_fn)(params, x, y_target, p_sigma)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    n_qubit = np.shape(pp_Y)[1]
    
    # Storage for parameters and losses during training for animation
    params_history = []
    loss_history = []

    # Convert parameters to jnp arrays for JAX compatibility
    pp_phi = jnp.array(np.random.rand(n_qubit, depth))
    ppp_omega = jnp.array(np.random.rand(n_qubit, depth, depth))
    pp_alpha = jnp.array(np.random.rand(n_qubit, depth))

    # Initialize optimizer (Adam with learning rate)
    optimizer = optax.adam(learning_rate=lr)
    opt_state = optimizer.init((pp_phi, ppp_omega, pp_alpha))
    progress = tqdm.tqdm(range(num_epochs))
    
    params = (pp_phi, ppp_omega, pp_alpha)
    for epoch in progress:
        X = jnp.array(np.random.rand(num_imgs, depth))
        Yidx = np.random.choice(len(pp_Y), size=num_imgs, replace=False)
        Y = jnp.array(pp_Y[Yidx])
        params, opt_state, loss = train_step(params, opt_state, X, Y, p_sigma)
        loss_history.append(loss)
        params_history.append(params)
        progress.set_description(f"MMD = {loss:.6f}")
        
    generated = None
    if generate:
        x = jnp.array(np.random.rand(len(pp_Y), depth))
        generated = jv_multi_Uuat(x, pp_phi, ppp_omega, pp_alpha)


    return params, params_history, loss_history, generated