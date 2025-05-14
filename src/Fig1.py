"""
Creates the inset of Fig1a
It simulates the evolution of occupation numbers and capital moments starting close at a
steady state. Then plots the final occupation number distributions.
"""

from sigmoidal_transition_prob import Economic_Process_sigmoidal
import numpy as np
import plot_libarry as plt_lib

if __name__ == "__main__":
    tau = 300
    num_levels = 5
    num_of_agents = 500**2
    beta = 15
    processes = []
    for i in [1,2]:
        new_ivp = np.loadtxt(f"data/steady_state_{i}.txt")
        temp_process = Economic_Process_sigmoidal(
            number_of_agents=num_of_agents,
            number_of_levels=num_levels,
            update_time=tau,
            temperature=1 / beta,
            initial_occupation_numbers=new_ivp[0],
            initial_capital_moments=new_ivp[1:],
            labour=num_of_agents,
            elasticities=[0.5, 0.5],
            production_constant=1,
            depreciation=0.05,
            exploration=0.05,
            gamma=1,
        )
        #solve ode to ensure steady state
        temp_process.solve_economics(5000)
        processes.append(temp_process)
    plt_lib.plot_saving_rate_distribution_from_procces(processes)
