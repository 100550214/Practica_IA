# Import required dependencies
import argparse
import json
from pathlib import Path

import numpy as np

from ControlModule import ControlModule
from DemandGenerator import generate_demand
from Metrics import *
from Plotter import *
from Reactor import Reactor

def get_args() -> tuple[Reactor, float, int]:
    # Collect the reactor configuration and controller parameters from the CLI.
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-reactor", "-i", type=str, default="Reactors/R0.json", help="Path of the reactor's JSON file")
    parser.add_argument("--gamma", "-g", type=float, default=0.9, help="Discount factor used in the MDP")
    parser.add_argument("--random-seed", "-r", type=int, default=42, help="Pseudo-random number generator seed")

    args = parser.parse_args()

    # Resolve relative reactor paths from this script's directory so the program
    # can be launched from any working directory.
    input_reactor_path = Path(args.input_reactor)

    if not input_reactor_path.is_absolute():
        input_reactor_path = Path(__file__).resolve().parent / input_reactor_path

    if not input_reactor_path.is_file():
        raise FileNotFoundError(
            f"No se encontró el archivo del reactor: {input_reactor_path}\n"
            f"Verifica que exista el archivo o pasa una ruta válida con --input-reactor."
        )

    # Echo the selected setup before the simulation starts.
    print(f"Loading reactor from file: {input_reactor_path}")
    print(f"Using gamma (discount factor): {args.gamma}")
    print(f"Using {args.random_seed} as random seed")

    # Load and validate the JSON fields needed to build a Reactor object.
    with open(input_reactor_path, 'r', encoding='utf-8') as file:
        json_data = json.load(fp=file)

        required_fields = (
            "model",
            "effective_section",
            "neutron_flux",
            "core_volume",
            "fision_energy",
            "probabilities",
        )
        missing_fields = [field for field in required_fields if field not in json_data]
        if missing_fields:
            raise ValueError(
                f"Error: reactor file '{input_reactor_path}' is missing required fields: {', '.join(missing_fields)}"
            )

        reactor = Reactor(model=json_data['model'],
                          effective_section=float(json_data['effective_section']),
                          neutron_flux=float(json_data['neutron_flux']),
                          core_volume=float(json_data['core_volume']),
                          fision_energy=float(json_data['fision_energy']),
                          probabilities=dict(json_data['probabilities']))

    # Show the loaded reactor parameters using Reactor.__str__.
    print(reactor)  # Overloaded in the __str__ method of Reactor's class

    return reactor, args.gamma, args.random_seed

def main() -> None:
    # Parse the simulation setup.
    reactor, gamma, random_seed = get_args()

    if not 0.0 < gamma < 1.0:
        raise ValueError("Error: gamma must be strictly between 0 and 1.")

    # Make the stochastic demand and reactor transitions reproducible.
    np.random.seed(random_seed)

    # Extract action probabilities in the order expected by the radar plot.
    probs = reactor.probabilities
    probs_array = np.array([probs['decrease'],
                            probs['maintain'],
                            probs['increase']], dtype=np.float64)
    
    # Make a radar-plot with the reactor probabilities
    plot_reactor_as_radar(probs=probs_array)
    
    # Generate the normalized target power signal that the controller tracks.
    demand = generate_demand(n_samples=512)

    # The MDP discretizes normalized power into 100 states and uses three
    # possible actions: decrease, maintain, and increase.
    n_states  = 100
    n_actions = 3

    # Run the controller to produce the reactor response for the demand signal.
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
