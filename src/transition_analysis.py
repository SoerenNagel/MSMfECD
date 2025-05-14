import numpy as np
from src.sigmoidal_transition_prob import Economic_Process_sigmoidal
from src.phase_space_functions import (
    transition_rates,
    wages,
    return_on_investment,
)

def mean_capital_change_during_spike(solution, times, level):
    """
    Compute the mean change in capital and consumption during a specified time interval.

    This function integrates the capital growth rate over the given time interval
    to determine the total change in capital. It assumes a solver step size of 1.
    
    Parameters:
    -----------
    solution : Economic_Process
        A solved instance of the economic process containing capital dynamics.
    times : list of int
        A two-element list specifying the start and end indices of the time interval.
    level : int
        The index specifying the capital level of interest.

    Returns:
    --------
    mean_capital_growth : float
        The integrated capital change over the specified time interval.
    mean_consumption_growth : float
        The integrated consumption change over the specified time interval.
    """
    s_l = solution.saving_rates[level]
    moments = np.array(solution.capital_moments[times[0] : times[1]])
    n = solution.unpacking_occupation_numbers()[times[0] : times[1]]
    r = np.array(
        [
            return_on_investment(
                moments[i][0],
                solution.production_constant,
                solution.elasticities,
                solution.labour,
                n[i],
            )
            for i in range(times[1] - times[0])
        ]
    )
    # data for wages plot
    wage = np.array(
        [
            wages(
                n[i],
                moments[i][0],
                solution.production_constant,
                solution.elasticities,
                solution.labour,
            )
            for i in range(times[1] - times[0])
        ]
    )
    dm1dt = (r * s_l - solution.depreciation * np.ones(times[1] - times[0])) * moments[
        :, 0, level
    ] + s_l * wage * solution.labour / solution.number_of_agents
    mean_capital_growth = np.trapz(dm1dt)

    mean_consumption_growth = (1 - s_l) * np.trapz(
        r
        * (
            (r * s_l - solution.depreciation * np.ones(times[1] - times[0]))
            * moments[:, 0, level]
            + s_l * wage * solution.labour / solution.number_of_agents
        )
    )

    return mean_capital_growth, mean_consumption_growth


def transition_data(solution, times, transitions):
    """
    Compute the number of agents transitioning between saving rate levels 
    in a given time interval.

    This function calculates transition rates between saving rate levels 
    and integrates them over the specified time interval. It assumes that 
    wages and returns remain approximately constant within the interval.

    Parameters:
    -----------
    solution : Economic_Process
        A solved instance of the economic process containing capital and agent dynamics.
    times : array-like, shape (2,)
        A two-element array specifying the start and end indices of the time interval.
    transitions : array-like, shape (N, 2)
        A list of transitions of interest, where each element (k, l) represents 
        a transition from saving rate level k to level l.
    """
    # finding indexes, for the specified times
    n = solution.unpacking_occupation_numbers()[times[0] : times[1]]
    moments = np.array(solution.capital_moments[times[0] : times[1]])
    # will get the structure t_tares[time][k][l] is the rate from level k to level l
    t_rates = np.zeros(
        (times[1] - times[0], solution.number_of_levels, solution.number_of_levels)
    )
    for i in range(times[1] - times[0]):
        t_rates[i] = transition_rates(
            moments[i][0],
            moments[i][1],
            moments[i][2],
            n[i],
            solution.update_time,
            solution.exploration,
            solution.number_of_levels,
            solution.production_constant,
            solution.elasticities,
            solution.labour,
            solution.saving_rates,
            solution.number_of_agents,
            solution.temperature,
        )
    number_of_transitions = (
        solution.number_of_agents / solution.update_time
    ) * np.ones(times[1] - times[0])
    for i in range(times[1] - times[0]):
        for l in range(solution.number_of_levels):
            number_of_transitions[i] += -t_rates[i][l][l]

    concentrations = np.zeros(len(transitions))
    mean_capital_influx = np.zeros((len(transitions)))
    mean_consumption_fluxes = np.zeros((len(transitions)))
    r = np.array(
        [
            return_on_investment(
                moments[i][0],
                solution.production_constant,
                solution.elasticities,
                solution.labour,
                n[i],
            )
            for i in range(times[1] - times[0])
        ]
    )

    for i, transition in enumerate(transitions):
        k, l = transition
        concentrations[i] = np.trapz(t_rates[:, k, l])
        mean_capital_influx[i] = np.trapz(
            t_rates[:, k, l] * (moments[:, 0, k] - moments[:, 0, l]) / n[:, l]
        )
        mean_consumption_fluxes[i] = (1 - solution.saving_rates[l]) * np.trapz(
            r * t_rates[:, k, l] * (moments[:, 0, k] - moments[:, 0, l]) / n[:, l]
        )

    return concentrations, mean_capital_influx, mean_consumption_fluxes
