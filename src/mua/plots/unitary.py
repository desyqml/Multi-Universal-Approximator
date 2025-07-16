import numpy as np
import pennylane as qml
from matplotlib.pyplot import savefig

from ..unitaries.Uuat import Uuat, ansatz


def show_single(depth, save):
    """
    Visualize a single quantum circuit.

    Parameters
    ----------
    depth : int
        Number of layers in the circuit.
    save : str
        Path to save the figure to. If empty, the figure is not saved.

    Returns
    -------
    None
    """
    p_X = np.array([0])  # Input to the circuit
    pp_w = np.array([1] * depth)  # Weights for each layer
    p_alpha = np.arange(depth)  # Alpha parameters for each layer
    p_phi = np.arange(depth)  # Phi parameters for each layer

    # Draw the quantum circuit
    qml.draw_mpl(Uuat)(p_X, pp_w, p_alpha, p_phi)

    # Save the figure if a save path is provided
    if save:
        savefig(save)

def show_multi(num_qubits, depth, save):
    """
    Visualize a multi-qubit quantum circuit.

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    depth : int
        Number of layers in the circuit.
    save : str
        Path to save the figure to. If empty, the figure is not saved.

    Returns
    -------
    None
    """
    p_X = np.array([0])  # Input to the circuit
    pp_w = np.array([1] * depth)  # Weights for each layer
    p_alpha = np.arange(depth)  # Alpha parameters for each layer
    p_phi = np.arange(depth)  # Phi parameters for each layer

    # Initialize the quantum device
    device = qml.device('default.qubit', wires=num_qubits)

    @qml.qnode(device)
    def multi_Uuat(p_X, pp_w, p_alpha, p_phi):
        # Apply the ansatz to each qubit
        for i in range(num_qubits):
            ansatz(p_X, p_phi, pp_w, p_alpha, wire=i)
        
        # Return the expectation value of PauliZ for each qubit
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    # Draw the quantum circuit
    qml.draw_mpl(multi_Uuat)(p_X, pp_w, p_alpha, p_phi)

    # Save the figure if a save path is provided
    if save:
        savefig(save)
        savefig(save)
