import jax.numpy as jnp
import pennylane as qml

from ..utils import dev1


def ansatz(p_x, p_phi, pp_w, p_alpha, wire=0):
    def U(p_x, phi, p_w, alpha, wire):
        qml.RZ(2 * (jnp.dot(p_w, p_x) + alpha), wires=wire)
        qml.RY(2 * phi, wires=wire)

    for phi, p_w, alpha in zip(p_phi, pp_w, p_alpha):
        U(p_x, phi, p_w, alpha, wire)


@qml.qnode(dev1, interface="jax")
def Uuat(x, p_phi, pp_w, p_alpha):
    ansatz(x, p_phi, pp_w, p_alpha)
    return qml.expval(qml.PauliZ(0))
