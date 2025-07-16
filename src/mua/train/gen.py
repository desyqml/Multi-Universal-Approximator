# train_mmd.py
from pathlib import Path
from typing import Callable, Dict

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import tqdm
from jax import jit, value_and_grad, vmap

from ..losses import mmd
from ..unitaries.multiUuat import multi_Uuat
from ..utils import COLORS, median_dist

jv_multi_Uuat = jit(vmap(multi_Uuat, in_axes=(0, None, None, None)))


def random_noise(num_vecs: int, num_qubits: int) -> jnp.ndarray:
    """Uniform noise ∈ [0, 1). Shape = (num_vecs, num_qubits)."""
    return jnp.array(np.random.rand(num_vecs, num_qubits))


def train(
    pp_Y: jnp.ndarray,
    depth: int,
    num_epochs: int = 200,
    num_imgs: int = 64,
    lr: float = 1e-1,
    p_sigma: list[float] | None = None,
    generate: int = 0,
    frequency: int = 1,
    save: str | Path = "",
    other_metrics: list[Callable] = [],
):
    """
    Parameters
    ----------
    pp_Y         : (num_samples, num_qubits) numpy array of real data vectors.
    depth        : circuit depth.
    num_epochs   : how many full dataset passes to train.
    num_imgs   : mini-batch size (last batch is dropped if smaller to
                   keep shapes static for JIT).
    lr           : Adam learning rate.
    p_sigma      : bandwidth list for the MMD kernel.  If None → median rule.
    generate     : if >0, generate this many fake samples with final model.
    frequency    : evaluate `other_metrics` every `frequency` epochs.
    save         : directory where loss curve PDF is written.
    other_metrics: list of callables metric(fake, real) → float.
    """
    save = Path(save)
    save.mkdir(parents=True, exist_ok=True)

    # ---------------------------- data stats --------------------------
    num_samples, num_qubits = pp_Y.shape
    num_batches = num_samples // num_imgs  # drop remainder for shape stability

    # ------------------------ model parameters ------------------------
    key_shape = (num_qubits, depth)
    pp_phi = jnp.array(np.random.rand(*key_shape))
    ppp_omega = jnp.array(np.random.rand(num_qubits, depth, depth))
    pp_alpha = jnp.array(np.random.rand(*key_shape))

    params = (pp_phi, ppp_omega, pp_alpha)

    # ---------------------------- optimiser --------------------------
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(params)

    # ------------------------ kernel bandwidth -----------------------
    if p_sigma is None:
        sigma = median_dist(pp_Y, 1000)
        p_sigma = [k * sigma for k in [0.1, 0.5, 1, 2, 5, 10]]

    # ------------------------- loss function -------------------------
    def loss_fn(params, x, y_target, p_sigma):
        pp_phi, ppp_omega, pp_alpha = params
        y_fake = jv_multi_Uuat(x, pp_phi, ppp_omega, pp_alpha)
        loss_val = mmd(y_fake, y_target, p_sigma)
        return loss_val, y_fake

    @jit
    def train_step(params, opt_state, x, y_target, p_sigma):
        (loss_val, aux), grads = value_and_grad(loss_fn, has_aux=True)(
            params, x, y_target, p_sigma
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss_val, aux  # aux == fake_y

    # --------------------------- bookkeeping -------------------------
    loss_history: list[float] = []
    params_history: list[tuple] = []

    metrics: Dict[str, Dict] = {
        m.__name__: {"epoch": [], "value": []} for m in other_metrics
    }

    progress = tqdm.tqdm(range(num_epochs), desc="Training", ncols=90)

    for epoch in progress:
        # shuffle once per epoch
        perm = np.random.permutation(num_samples)
        epoch_loss = 0.0

        for b in range(num_batches):
            idx = perm[b * num_imgs : (b + 1) * num_imgs]
            Y_batch = jnp.array(pp_Y[idx])
            X_batch = random_noise(num_imgs, depth)

            params, opt_state, loss_val, fake_y = train_step(
                params, opt_state, X_batch, Y_batch, p_sigma
            )
            epoch_loss += float(loss_val) * num_imgs  # accumulate

        epoch_loss /= num_samples
        loss_history.append(epoch_loss)
        params_history.append(params)

        # auxiliary metrics
        if (epoch % frequency) == 0:
            for m in other_metrics:
                val = float(m(fake_y, Y_batch))
                metrics[m.__name__]["epoch"].append(epoch)
                metrics[m.__name__]["value"].append(val)

        desc = f"MMD={epoch_loss:.6f}"
        for m in other_metrics:
            desc += f", {m.__name__}={metrics[m.__name__]['value'][-1]:.6f}"
        progress.set_description(desc)

    generated = None
    if generate:
        X_gen = random_noise(generate, depth)
        generated = jv_multi_Uuat(X_gen, *params)

    plt.figure(figsize=(6, 3))
    plt.xlabel("Epoch")
    plt.ylabel("Losses")
    plt.plot(loss_history, color=COLORS["mmd"], label="MMD")
    for m in other_metrics:
        plt.plot(
            metrics[m.__name__]["epoch"],
            metrics[m.__name__]["value"],
            "--",
            color=COLORS.get(m.__name__, "gray"),
            label=m.__name__,
        )
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(save / f"loss_clic_q{num_qubits}_d{depth}.pdf")

    return params, params_history, loss_history, generated
