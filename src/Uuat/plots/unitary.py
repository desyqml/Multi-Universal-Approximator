import numpy as np
import pennylane as qml
from matplotlib.pyplot import savefig

from ..unitaries.Uuat import Uuat, ansatz


def show_single(depth, save):
    p_X = np.array([0])
    pp_w = np.array([1]*depth)
    p_alpha = np.arange(depth)
    p_phi = np.arange(depth)
    
    qml.draw_mpl(Uuat)(p_X, pp_w, p_alpha, p_phi)

    if save:
        savefig(save)

def show_multi(num_qubits, depth, save):
    p_X = np.array([0])
    pp_w = np.array([1]*depth)
    p_alpha = np.arange(depth)
    p_phi = np.arange(depth)
    
    device = qml.device('default.qubit', wires=num_qubits)
    @qml.qnode(device)
    def multi_Uuat(p_X, pp_w, p_alpha, p_phi):
        for i in range(num_qubits):
            ansatz(p_X, p_phi, pp_w, p_alpha, wire = i)
            
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]
    
    qml.draw_mpl(multi_Uuat)(p_X, pp_w, p_alpha, p_phi)

    if save:
        savefig(save)