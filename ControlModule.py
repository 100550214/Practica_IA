# Import required dependencies
import numpy as np
import mdptoolbox

class ControlModule:
    def __init__(self):
        """Dummy constructor to use the class as a namespace for static methods."""
        pass

    @staticmethod
    def generate_P(n_states: np.int32, n_actions: np.int32, n_probs: dict) -> np.ndarray:
        """Generate the MDP transition matrix for every action and power state."""
        if n_states < 2:
            raise ValueError("Error: the controller requires at least two power states.")
        if n_actions != 3:
            raise ValueError("Error: this controller expects exactly three actions: decrease, maintain and increase.")

        # Each action must provide the probabilities for its three possible
        # stochastic outcomes, and each distribution must be valid.
        for action in ("decrease", "maintain", "increase"):
            if action not in n_probs:
                raise ValueError(f"Error: missing '{action}' probabilities in reactor configuration.")

            probs = np.asarray(n_probs[action], dtype=np.float64)
            if probs.shape != (3,):
                raise ValueError(f"Error: '{action}' must define exactly three transition probabilities.")
            if np.any(probs < 0.0):
                raise ValueError(f"Error: '{action}' contains negative transition probabilities.")
            if not np.isclose(probs.sum(), 1.0):
                raise ValueError(f"Error: '{action}' probabilities must sum to 1.0.")

        tmatrix = np.zeros((n_actions, n_states, n_states))

        # Vector of current states. The transition matrix is indexed as
        # [action, current_state, next_state].
        i = np.arange(n_states)

        # Decrease can move the power down two states, down one state, or keep
        # it unchanged. Boundary states saturate at 0.
        if "decrease" in n_probs:
            p = n_probs["decrease"]
            tmatrix[0, i, np.maximum(i - 2, 0)] += p[0]
            tmatrix[0, i, np.maximum(i - 1, 0)] += p[1]
            tmatrix[0, i, i] += p[2]

        # Maintain still allows small stochastic movement around the current
        # state, because the reactor model is not perfectly deterministic.
        if "maintain" in n_probs:
            p = n_probs["maintain"]
            tmatrix[1, i, np.maximum(i - 1, 0)] += p[0]
            tmatrix[1, i, i] += p[1]
            tmatrix[1, i, np.minimum(i + 1, n_states - 1)] += p[2]

        # Increase mirrors the decrease action and saturates at the maximum
        # power state.
        if "increase" in n_probs:
            p = n_probs["increase"]
            tmatrix[2, i, i] += p[0]
            tmatrix[2, i, np.minimum(i + 1, n_states - 1)] += p[1]
            tmatrix[2, i, np.minimum(i + 2, n_states - 1)] += p[2]

        return tmatrix

    @staticmethod
    def generate_R(n_states: np.int32, n_actions: np.int32, demand: np.int64) -> np.ndarray:
        """Generate the tracking-cost matrix for one demand value."""
        cmatrix = np.zeros((n_actions,n_states,n_states))
        scale = n_states - 1
        i = np.arange(n_actions)
        j = np.arange(n_states)
        k = np.arange(n_states)
        I,J,K = np.meshgrid(i,j,k,indexing='ij')
        current = J/scale
        next_ = K/scale
        decrease = (I==0)
        maintain = (I==1)
        increase = (I==2)

        # Penalize actions that move away from the demand or refuse to move
        # when the current state is still below/above the target.
        penalize = ((current < demand) & decrease & (current < next_)) | \
                   ((current > demand) & increase & (current > next_)) | \
                   ((current < demand) & maintain & (current == next_)) | \
                   ((current > demand) & maintain & (current == next_)) | \
                   ((current == demand) & decrease & (current < next_)) | \
                   ((current == demand) & increase & (current > next_))

        # Base cost is the distance between the demand and the possible next
        # state; counterproductive moves receive double cost.
        cost = np.abs(demand-next_)
        cmatrix[:] = np.where(penalize,cost*2,cost)
        return cmatrix


    @staticmethod
    def control_iteration(current_power: np.int32,
                          tmatrix: np.ndarray,
                          cmatrix: np.ndarray,
                          discount: np.float64) -> np.float64:
        """Solve the MDP for the current state and return the best action."""
        if not 0.0 < discount < 1.0:
            raise ValueError("Error: gamma must be strictly between 0 and 1.")
        # PyMDPToolbox maximizes rewards, so we negate the tracking-cost matrix.
        result = mdptoolbox.mdp.ValueIteration(tmatrix, -cmatrix, discount)
        result.run()
        return result.policy[current_power]

    @staticmethod
    def control_loop(demand: np.ndarray,
                     probs: dict,
                     n_states: np.int32,
                     n_actions: np.int32,
                     gamma: np.float64) -> np.ndarray:
        """Run the controller over the full demand signal."""
        tmatrix = ControlModule.generate_P(n_states,n_actions,probs)
        state_scale = n_states - 1
        current_power = np.int32(0)
        power_curve = []
        for i in range(len(demand)):
            # Candidate next states for each action from the current state.
            decrease = [max(0,current_power-2),max(0,current_power-1),current_power]
            maintain = [max(0,current_power-1),current_power,min(n_states-1,current_power+1)]
            increase = [current_power,min(n_states-1,current_power+1),min(n_states-1,current_power+2)]

            # Recompute costs for the current demand sample, choose an optimal
            # action, then sample the reactor's stochastic response.
            cmatrix = ControlModule.generate_R(n_states,n_actions,demand[i])
            optimized_action = ControlModule.control_iteration(current_power,tmatrix,cmatrix,gamma)
            if optimized_action == 0:
                current_power = np.random.choice(decrease,p=probs["decrease"])
            elif optimized_action == 1:
                current_power = np.random.choice(maintain,p=probs["maintain"])
            elif optimized_action == 2:
                current_power = np.random.choice(increase,p=probs["increase"])
            power_curve.append(current_power / state_scale)

        return np.array(power_curve)

