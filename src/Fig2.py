from sigmoidal_transition_prob import Economic_Process_sigmoidal
import numpy as np
import plot_libarry as plt_lib

if __name__ == "__main__":
    tau = 300
    mean = 100
    var = 30
    sdev = np.sqrt(var)
    skewness = 0
    third_mom = sdev**3 * skewness + 3 * mean * sdev**2 + mean**3
    num_levels = 5
    num_of_agents = 500
    initial_moments = np.array([
        [mean for _ in range(num_levels)],
        [var for _ in range(num_levels)],
        [third_mom for _ in range(num_levels)],
    ])

    beta = 5
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

    process.solve_SDE_economics(20_000,1)
    plt_lib.plot_transition_rates(process)
    
