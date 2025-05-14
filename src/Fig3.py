"""
This script generates the Figures used in Fig.3 of the main paper.
To get the proper scaling of teh capital returns, zoom into the corresponding figures first.
It may be the case, that there is no transition in the time interval.
In that case, just rerun the script.
"""
import os
import sys
import numpy as np


# needed for the imports of our packages
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Change the working directory to the script to ensure correct paths
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

import numpy as np
from src import Economic_Process_sigmoidal
from src import plt_lib
from src import transition_data, mean_capital_change_during_spike

if __name__ == "__main__":
    tau = 300
    mean = 100
    var = 30
    sdev = np.sqrt(var)
    skewness = 0
    third_mom = sdev**3 * skewness + 3 * mean * sdev**2 + mean**3
    num_levels = 5
    # due to scaling of the diffusion, this
    num_of_agents = 500
    initial_moments = np.array([
        [mean for _ in range(num_levels)],
        [var for _ in range(num_levels)],
        [third_mom for _ in range(num_levels)],
    ])

    beta = 15
    process = Economic_Process_sigmoidal(
        number_of_agents=num_of_agents,
        number_of_levels=num_levels,
        update_time=tau,
        temperature=1 / beta,
        initial_occupation_numbers=np.full(num_levels, num_of_agents / num_levels),
        initial_capital_moments=initial_moments,
        labour=num_of_agents,
        elasticities=[0.5, 0.5],
        production_constant=1,
        depreciation=0.05,
        exploration=0.05,
        gamma=1,
    )
    process.solve_SDE_economics(15_000, 1)
    plt_lib.plot_skewness_illustration_5_level(process)
    # plt_lib.plot_simple_time_series(process)
    times = input(
        "Enter the time steps, where spike Consumption spike starts and where it ends,"
        "i.e. times corresponding to the pink box in Fig3(e) (separated by ',': t1,t2):"
    )
    times = list(times.split(","))
    times = [eval(i) for i in times]
    
    #get the capital and consumtion fluxes into the level s_l = 0.27
    # the fluxes only contain the change due to switching agents
    agent_flux, capital_flux, consum_fluxes = transition_data(
        process, times, [(m, 1) for m in [0, 2, 3, 4]]
    )
    print("Concentration Fluxes", agent_flux * process.number_of_agents**-1)
    print("Capital Fluxes", capital_flux)
    print("Consumption Fluxes", consum_fluxes)
    #Now we get the changes of consumption and mean capital due to the time evolution
    # within the levels and without changing agents.
    mean_capital_growth, mean_consum_growth = mean_capital_change_during_spike(
        process, times, level=1
    )
    print("Growth of Mean Capital due to economic dynamics", mean_capital_growth)
    print("Growth of Mean Consumption due to economic dynamics", mean_consum_growth)
    dKf = np.sum(capital_flux) + mean_capital_growth #total change in capital
    print("Total Change of Mean Capital during the Spike (in level 2)", dKf)

    print(
        "Total Change of Mean Consumption during the Spike (in level 2)",
        np.sum(consum_fluxes) + mean_consum_growth,
    )

    cap_fluxes = list(capital_flux)
    cap_fluxes.append(mean_capital_growth)
    cap_fluxes = np.array(cap_fluxes)
    plt_lib.plot_fluxes(cap_fluxes)

    con_fluxes = list(consum_fluxes)
    con_fluxes.append(mean_consum_growth)
    # relative
    con_fluxes = np.array(con_fluxes)
    plt_lib.plot_fluxes(con_fluxes)
