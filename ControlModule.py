# Import required dependencies
import numpy as np
import mdptoolbox

class ControlModule:
    def __init__(self):
        """ Dummy constructor to use the Python Class as a namespace """
        pass

    @staticmethod
    def generate_P(n_states: np.int32, n_actions: np.int32, n_probs: np.ndarray) -> np.ndarray:
        """ Function that generates the probabilities (transition) matrix """
        
        tmatrix = np.zeros((n_actions, n_states, n_states))

        i = np.arange(n_states)

        if "decrease" in n_probs:
            p = n_probs["decrease"]
            tmatrix[0, i, np.maximum(i - 2, 0)] += p[0]
            tmatrix[0, i, np.maximum(i - 1, 0)] += p[1]
            tmatrix[0, i, i] = p[2]

        if "maintain" in n_probs:
            p = n_probs["maintain"]
            tmatrix[1, i, np.maximum(i - 1, 0)] += p[0]
            tmatrix[1, i, i] = p[1]
            tmatrix[1, i, np.minimum(i + 1, n_states - 1)] += p[2]

        if "increase" in n_probs:
            p = n_probs["increase"]
            tmatrix[2, i, i] = p[0]
            tmatrix[2, i, np.minimum(i + 1, n_states - 1)] += p[1]
            tmatrix[2, i, np.minimum(i + 2, n_states - 1)] += p[2]

        return tmatrix

    @staticmethod
    def generate_R(n_states: np.int32, n_actions: np.int32, demand: np.ndarray) -> np.ndarray:
        """ Function that generates the rewards (costs) matrix """
        cmatrix = np.zeros((n_actions,n_states,n_states))
        for i in range (n_actions):
            for k in range(n_states):
                for z in range (n_states):
                    d1 = abs(demand-k/100)
                    d2 = abs(demand-z/100)
                    if d2 > d1:
                        cmatrix[i][k][z] = d2*2
                    else:
                        cmatrix[i][k][z] = d2
        
        return cmatrix
                    

    @staticmethod
    def control_iteration() -> np.int32:
        """ Function that computes one control-iteration """
        ### TO BE COMPLETED BY THE STUDENTS ###
        ...

    @staticmethod
    def control_loop(demand: np.ndarray, 
                     probs: np.ndarray,
                     n_states: np.int32, 
                     n_actions: np.int32,
                     gamma: np.float64) -> np.ndarray:
        """ Function that computes all the required iterations (control-loop) to satisfy the power demand """
        ### TO BE COMPLETED BY THE STUDENTS ###

        ### DUMMY BEHAVIOUR TO PREVENT CRASHING (MUST BE DELETED AFTER THE FULL IMPLEMENTATION) ###
        return np.zeros_like(a=demand, dtype=np.float64)
        ### ###
