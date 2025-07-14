import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML
from matplotlib.animation import FuncAnimation

from ..train.regression import jv_Uuat, x_regression


def animated(target_fun, params_history, save='', max_frames=None):
    target_vals = target_fun(x_regression)
    fig, ax = plt.subplots()
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_title('Training Progress of Quantum Circuit Output')
    ax.set_xlabel('x')
    ax.set_ylabel('output')

    line_pred, = ax.plot([], [], label='Predicted', color='blue')
    ax.plot(x_regression, target_vals, label='Target', color='orange')
    text_loss = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    ax.legend(loc='upper right')

    # pick frames from params_history
    frames = (np.linspace(0, len(params_history) - 1, max_frames, dtype=int)
              if max_frames is not None else range(len(params_history)))

    def init():
        line_pred.set_data([], [])
        text_loss.set_text('')
        return line_pred, text_loss

    def update(epoch_idx):
        p_omega, p_alpha, p_phi = params_history[epoch_idx]
        y_pred = jv_Uuat(x_regression, p_omega, p_alpha, p_phi)
        line_pred.set_data(x_regression, y_pred)
        text_loss.set_text(f'Epoch: {epoch_idx}')
        return line_pred, text_loss

    anim = FuncAnimation(fig, update, frames=frames,
                         init_func=init, blit=True, interval=100)

    if save:
        writer = 'pillow' if save.endswith('.gif') else 'ffmpeg'
        anim.save(save, writer=writer)

    return HTML(anim.to_jshtml())
