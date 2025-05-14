import numpy as np
from sigmoidal_transition_prob import Economic_Process_sigmoidal
import plot_libarry as plt_lib
import time
import pickle
import tqdm


def run_SDE_solver(agents, tau, num_levels, target_time, temperature):
    for num_of_agents in agents:
        mean = 100
        var = 30
        sdev = np.sqrt(var)
        skewness = 0
        third_mom = sdev**3 * skewness + 3 * mean * sdev**2 + mean**3
        initial_moments = np.array([
            [mean for _ in range(num_levels)],
            [var for _ in range(num_levels)],
            [third_mom for _ in range(num_levels)],
        ])

        process = Economic_Process_sigmoidal(
            number_of_agents=num_of_agents,
            number_of_levels=num_levels,
            update_time=tau,
            temperature=temperature,
            initial_occupation_numbers=np.full(num_levels, num_of_agents / num_levels),
            initial_capital_moments=initial_moments,
            labour=num_of_agents,
            elasticities=[0.5, 0.5],
            production_constant=1,
            depreciation=0.05,
            exploration=0.05,
            gamma=1,
        )
        # file = f"tau={tau}, Agents {num_of_agents}, Temp = {temperature}Levels={num_levels} 2nd_run"
        # with open(file, 'rb') as f:
        #    process = pickle.load(f)
        time_interval = 500
        process.solve_SDE_economics(time_interval, 1)
        # plt_lib.plot_simple_time_series(process)
        times = np.linspace(time_interval, target_time, 60)
        start_time = time.time()
        for c_time in tqdm.tqdm(times):
            process.continue_solving(time_interval, 1)
            run_time = np.round(time.time() - start_time)
            if run_time > 24 * 3600:
                cont = int(input("Continue? yes = 1 : no = 0"))
                if cont == 0:
                    break
                elif cont == 1:
                    start_time = time.time()
        print(("Run Time: ", np.round(time.time() - start_time) / (3600)), " Hours")


def continue_from_file(file, delete_steps):
    with open(file, "rb") as f:
        process = pickle.load(f)
    process.delete_tail_of_solution(delete_steps)
    process.continue_solving(delete_steps, 1)
    time_interval = 10_000
    # plt_lib.plot_simple_time_series(process)
    times = np.linspace(time_interval, target_time, 2)
    start_time = time.time()
    print("here")
    for c_time in tqdm.tqdm(times):
        print("here")
        process.continue_solving(time_interval, 1)
        run_time = np.round(time.time() - start_time)
        if run_time > 20 * 3600:
            cont = int(input("Continue? yes = 1 : no = 0"))
            if cont == 0:
                break
    plt_lib.plot_single_time_series_occupation_numbers(process)
    print(("Run Time: ", np.round(time.time() - start_time) / (3600)), " Hours")
    file = (
        "tau="
        + str(tau)
        + ", Agents "
        + str(num_of_agents)
        + ", Temp = "
        + str(process.temperature)
        + "Levels="
        + str(num_levels)
        + "CONTINUATION"
    )
    with open(file, "wb") as f:
        pickle.dump(process, f)


def plot_occupation_numbers(file):
    with open(file, "rb") as f:
        process = pickle.load(f)
    plt_lib.plot_single_time_series_occupation_numbers(process)
    plt_lib.plot_saving_rate_color_plot(process)


def main():
    agents = [2300]
    tau = 300
    num_levels = 30
    target_time = 50_000
    temperature = 0.02
    run_SDE_solver(agents, tau, num_levels, target_time, temperature)
    for num_of_agents in agents:
        file = f"tau={tau}, Agents {num_of_agents}, Temp = {temperature}Levels={num_levels} 2nd_run"
        # file = 'tau=' + str(tau) + ', Agents ' + str(num_of_agents) + ', Temp = ' + str(
        # temperature) + 'Levels=' + str(num_levels)

        # continue_from_file(file, 50_000)
        plot_occupation_numbers(file)
        # plt_lib.plot_simple_time_series(process, True)
        # plt_lib.plot_occupation_numbrs_over_capital_mean(process, mode='line')
