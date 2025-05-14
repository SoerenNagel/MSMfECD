import numpy as np
from sigmoidal_transition_prob import Economic_Process_sigmoidal
from phase_space_functions import production
import matplotlib.pyplot as plt
import tqdm
import plot_libarry as plt_lib
import concurrent.futures


def growth_run(pars):
    n0, beta, agent_numbers, delta_t = pars
    tau = 300
    mean = 100
    var = 30
    sdev = np.sqrt(var)
    skewness = 0
    third_mom = sdev**3 * skewness + 3 * mean * sdev**2 + mean**3
    num_levels = 5
    num_of_agents = n0
    initial_moments = [
        [mean for _ in range(num_levels)],
        [var for _ in range(num_levels)],
        [third_mom for _ in range(num_levels)],
    ]
    process = Economic_Process_sigmoidal(
        number_of_agents=num_of_agents,
        number_of_levels=num_levels,
        update_time=tau,
        temperature=1 / beta,
        initial_occupation_numbers=np.full(num_levels, num_of_agents / num_levels),
        initial_capital_moments=np.array(initial_moments),
        labour=num_of_agents,
        elasticities=[0.5, 0.5],
        production_constant=1,
        depreciation=0.05,
        exploration=0.05,
        gamma=1,
    )

    economic_production = np.array([])
    # let the system equilibriate
    process.solve_economics(1000)

    # main loop for simulation  with adiabaticly increasing agents
    for num_agents in tqdm.tqdm(agent_numbers):
        # ensure L/N = const
        process.number_of_agents = num_agents
        process.labour = num_agents

        last_occup_nums = process.unpacking_occupation_numbers()[1]
        process.social_process[-1].occupation_numbers[-1] = (
            num_agents
            * process.social_process[-1].occupation_numbers[-1]
            / np.sum(last_occup_nums)
        )
        process.continue_solving(delta_t, 1)
        if (
            num_agents == agent_numbers[-1]
            and process.unpacking_occupation_numbers()[-1][2] > 500
        ):
            plt_lib.plot_simple_time_series(process, True, True)
            aux_process = Economic_Process(
                number_of_agents=num_agents,
                number_of_levels=num_levels,
                update_time=tau,
                temperature=1 / beta,
                initial_occupation_numbers=process.unpacking_occupation_numbers()[-1]
                * (num_agents / n0),
                initial_capital_moments=initial_moments,
                labour=num_of_agents,
                elasticities=[0.5, 0.5],
                production_constant=1,
                depreciation=0.05,
                exploration=0.05,
                gamma=1,
            )
            aux_process.solve_economics(10_000)
            plt_lib.plot_simple_time_series(aux_process, True)

        moments = np.array(process.capital_moments[-delta_t:])
        n = process.unpacking_occupation_numbers()[-delta_t:]
        if np.isnan(np.sum(n)):
            return None
        prod = np.array(
            [
                production(
                    moments[i][0],
                    process.production_constant,
                    n[i],
                    process.elasticities,
                    process.labour,
                )
                for i in range(delta_t)
            ]
        )

        economic_production = np.concatenate([economic_production, prod])
    return economic_production


if __name__ == "__main__":
    beta = 35
    n0 = 500
    nt = 1800
    t = 40_000

    n = (t**-1) * np.log(nt**2 / n0**2)
    print(n)
    steps = 1000
    switching_times = np.linspace(0, t, steps)
    delta_t = int(switching_times[1] - switching_times[0])
    agent_numbers = n0 * np.exp(n * switching_times)

    with concurrent.futures.ProcessPoolExecutor() as executer:
        solutions = executer.map(
            growth_run, [(n0, beta, agent_numbers, delta_t) for _ in range(8)]
        )

    fig, axs = plt.subplots()
    axs2 = axs.twinx()
    # I have corrected the wrong SDE in the line below with an additional factor of N(t) at every time step

    axs2.plot(
        switching_times,
        agent_numbers * agent_numbers,
        ls="--",
        color="k",
        label=f"N(t) = exp(n_0t)",
    )
    for prod in solutions:
        if prod is not None:
            prod = prod * np.concatenate(
                [agent_num * np.ones(int(t * steps**-1)) for agent_num in agent_numbers]
            )
            window_size = 100
            i = 0

            moving_averages = []
            while i < len(prod) - window_size + 1:
                window = prod[i : i + window_size]

                window_average = np.round(np.sum(window) / window_size, 2)
                moving_averages.append(window_average)
                i += 1
            axs.plot(
                moving_averages,
                linewidth=1.2,
                color="tab:grey",
                alpha=0.7,
            )

    axs.set_ylabel(r"Economic Production $Y_t$")
    axs2.set_ylabel(r"$N_t=\exp(n_{0}t)$")
    axs.set_xlabel(r"$t$")
    axs.set_yscale("log")
    axs2.set_yscale("log")
    axs.legend()
    axs2.legend()

    axs.grid(True)
    plt.show()
