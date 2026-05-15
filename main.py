# Import required dependencies
import numpy as np
import mdptoolbox
import argparse
import json
from pathlib import Path
from Reactor import Reactor
from DemandGenerator import generate_demand
from Metrics import *
from Plotter import *


class ControlModule:
    def __init__(self):
        """ Dummy constructor to use the Python Class as a namespace """
        pass

    @staticmethod
    def generate_P(n_states: np.int32, n_actions: np.int32, n_probs: np.ndarray) -> np.ndarray:
        """ Function that generates the probabilities transition matrix """

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
    def generate_R(n_states: np.int32, n_actions: np.int32, demand: np.int64) -> np.ndarray:
        """ Function that generates the rewards costs matrix """

        cmatrix = np.zeros((n_actions, n_states, n_states))

        i = np.arange(n_actions)
        j = np.arange(n_states)
        k = np.arange(n_states)

        I, J, K = np.meshgrid(i, j, k, indexing="ij")

        current = J / n_states
        next_ = K / n_states

        decrease = I == 0
        maintain = I == 1
        increase = I == 2

        penalize = (
            ((current < demand) & decrease & (current < next_))
            | ((current > demand) & increase & (current > next_))
            | ((current < demand) & maintain & (current == next_))
            | ((current > demand) & maintain & (current == next_))
            | ((current == demand) & decrease & (current < next_))
            | ((current == demand) & increase & (current > next_))
        )

        cost = np.abs(demand - next_)
        cmatrix[:] = np.where(penalize, cost * 2, cost)

        return cmatrix

    @staticmethod
    def control_iteration(
        current_power: np.int32,
        tmatrix: np.ndarray,
        cmatrix: np.ndarray,
        discount: np.float64,
    ) -> np.float64:
        """ Function that computes one control iteration """

        result = mdptoolbox.mdp.ValueIteration(tmatrix, cmatrix, discount)
        result.run()

        return result.policy[current_power]

    @staticmethod
    def control_loop(
        demand: np.ndarray,
        probs: np.ndarray,
        n_states: np.int32,
        n_actions: np.int32,
        gamma: np.float64,
    ) -> np.ndarray:
        """ Function that computes all the required iterations control loop to satisfy the power demand """

        tmatrix = ControlModule.generate_P(n_states, n_actions, probs)

        current_power = np.int32(0)
        power_curve = []

        for i in range(len(demand)):
            decrease = [
                max(0, current_power - 2),
                max(0, current_power - 1),
                current_power,
            ]

            maintain = [
                max(0, current_power - 1),
                current_power,
                min(n_states - 1, current_power + 1),
            ]

            increase = [
                current_power,
                min(n_states - 1, current_power + 1),
                min(n_states - 1, current_power + 2),
            ]

            cmatrix = ControlModule.generate_R(n_states, n_actions, demand[i])
            optimized_action = ControlModule.control_iteration(
                current_power,
                tmatrix,
                cmatrix,
                gamma,
            )

            if optimized_action == 0:
                current_power = np.random.choice(decrease, p=probs["decrease"])
            elif optimized_action == 1:
                current_power = np.random.choice(maintain, p=probs["maintain"])
            elif optimized_action == 2:
                current_power = np.random.choice(increase, p=probs["increase"])

            power_curve.append(current_power / 100)

        return np.array(power_curve)

def get_args() -> tuple[Reactor, np.float64]:
    # Define the parser object
    parser = argparse.ArgumentParser()

    # Define the expected arguments to parse and their data types
    parser.add_argument("--input-reactor", "-i", type=str, default="Reactors", help="Path of the reactor's JSON file")
    parser.add_argument("--gamma", "-g", type=float, help="Discount factor used in the MDP")
    parser.add_argument("--random-seed", "-r", type=int, help="Pseudo-random number generator seed")

    # Parse the arguments
    args = parser.parse_args()

    # Convert the reactor path to a Path object
    input_reactor_path = Path(args.input_reactor)

    # If the path is relative, resolve it from the folder where main.py is located
    if not input_reactor_path.is_absolute():
        input_reactor_path = Path(__file__).resolve().parent / input_reactor_path

    # Check that the file exists before trying to open it
    if not input_reactor_path.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo del reactor: {input_reactor_path}\n"
            f"Verifica que exista el archivo o pasa una ruta válida con --input-reactor."
        )

    # Some verbose to check the correct parsing of the input arguments
    print(f"Loading reactor from file: {input_reactor_path}")
    print(f"Using gamma (discount factor): {args.gamma}")
    print(f"Using {args.random_seed} as random seed")

    # Build the Reactor object by reading the reactor's JSON file
    with open(input_reactor_path, 'r', encoding='utf-8') as file:
        json_data = json.load(fp=file)
        reactor   = Reactor(model=json_data['model'],
                            effective_section=float(json_data['effective_section']),
                            neutron_flux=float(json_data['neutron_flux']),
                            core_volume=float(json_data['core_volume']),
                            fision_energy=float(json_data['fision_energy']),
                            probabilities=dict(json_data['probabilities']))

    # Some verbose of the reactor loaded
    print(reactor)  # Overloaded in the __str__ method of Reactor's class

    # Return the Reactor object, gamma and the random seed
    return reactor, args.gamma, args.random_seed

def main() -> None:
    # Parse the main arguments
    reactor, gamma, random_seed = get_args()

    # Set the random seed
    np.random.seed(random_seed)

    # Get the probabilities from the reactor's dynamics
    probs = np.array([reactor.probabilities['decrease'], 
                      reactor.probabilities['maintain'], 
                      reactor.probabilities['increase']], dtype=np.float64)
    
    # Make a radar-plot with the reactor probabilities
    plot_reactor_as_radar(probs=probs)
    
    # Generate a random power demand
    demand = generate_demand(n_samples=512)

    # Define the number of MDP's states, actions and the discount factor (gamma)
    n_states  = 100
    n_actions = 3

    # Get the response time-series (answer to the demand time-series)
    response  = ControlModule.control_loop(demand=demand, 
                                           probs=probs,
                                           n_states=n_states,
                                           n_actions=n_actions,
                                           gamma=gamma)
    
    # Plot the original power demand
    plot_demand(demand=demand)

    # Plot the original power demand and the corresponding response
    plot_demand_response(demand=demand, response=response)

    # Plot the power response and the control bar percentaje employed
    plot_control_bars_usage(reactor=reactor, response=response)

    # Plot the correlation scatter-plot of both the demand and the response time-series
    plot_correlation(demand=demand, response=response)

    # Print the four regression metrics for the current demand-response data
    _MAE  = MAE(y_true=demand, y_pred=response)
    _MSE  = MSE(y_true=demand, y_pred=response)
    _R2   = R2(y_true=demand, y_pred=response)
    _Corr = Corr(y_true=demand, y_pred=response)
    print(f"MAE={_MAE:.6f}")
    print(f"MSE={_MSE:.6f}")
    print(f"R2={_R2:.6f}")
    print(f"Corr={_Corr:.6f}")

    # Plot the MAE and the MSE in a bar-plot
    plot_mae_and_mse(MAE=_MAE, MSE=_MSE)

    # Plot the R2 and the Corr in a bar-plot
    plot_r2_and_pearson(R2=_R2, Pearson=_Corr)

if __name__ == '__main__':
    main()
