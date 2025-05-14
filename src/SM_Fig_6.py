import matplotlib.pyplot as plt
import matplotlib
from sigmoidal_transition_prob import Economic_Process_sigmoidal
import plot_libarry as plt_lib
from scipy.stats import wasserstein_distance
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.interpolate import interp1d
from scipy.stats import entropy


def step_interp(interpolated_x, x, y, plot=False, no_shift=False):
    # Define step function using interp1d
    step_function = interp1d(x, y, kind="previous", fill_value="extrapolate")

    # Interpolate values
    if no_shift:
        interpolated_y = step_function(interpolated_x)
    else:
        interpolated_y = step_function(
            interpolated_x + 0.5 * np.diff(x)[0] * np.ones(len(interpolated_x))
        )

    if plot:
        fig, ax = plt.subplots()
        ax.plot(x, y, "bo", label="Original Data")
        ax.plot(
            interpolated_x, interpolated_y, "r-", label="Step Function Interpolation"
        )
        ax.set_yscale("log")
        plt.legend()
        plt.grid(True)
        plt.show()
        plt.close()
    return interpolated_y


if __name__ == "__main__":
    tau = 300
    mean = 100
    var = 30
    sdev = np.sqrt(var)
    skewness = 0
    third_mom = sdev**3 * skewness + 3 * mean * sdev**2 + mean**3
    num_levels = 5
    num_of_agents = 500**2

    processes = []
    max_level = 200
    levels = np.arange(4, max_level + 2, 1)
    plot_points = np.arange(10, max_level - 1, 20)
    if max_level not in plot_points:
        plot_points[-1] = max_level

    for level in levels:
        initial_moments = [
            [mean for _ in range(level)],
            [var for _ in range(level)],
            [third_mom for _ in range(level)],
        ]
        process = Economic_Process_sigmoidal(
            number_of_agents=num_of_agents,
            number_of_levels=level,
            update_time=tau,
            temperature=1 / 50,
            initial_occupation_numbers=np.full(level, num_of_agents / level),
            initial_capital_moments=initial_moments,
            labour=num_of_agents,
            elasticities=[0.5, 0.5],
            production_constant=1,
            depreciation=0.05,
            exploration=0.05,
            gamma=1,
        )
        processes.append(process)

    def run(process):
        process.solve_economics(4000)
        return process

    processes = Parallel(n_jobs=-1)(
        delayed(run)(process) for process in tqdm(processes)
    )
    print("Solving_Complete")

    ###################################################################################################################
    # Wasserstein direct
    ###################################################################################################################
    fig, ax = plt.subplots(2)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "custom",
        ["tab:blue", "tab:purple", "tab:red", "tab:orange", "gold"],
        len(processes),
    )
    approxs = []
    for i, process in tqdm(enumerate(processes)):
        n = process.unpacking_occupation_numbers()[-1]
        interpolation = np.interp(
            np.linspace(0.05, 0.95, max_level), process.saving_rates, n
        )
        interpolation = interpolation / np.trapz(
            interpolation, x=np.linspace(0.05, 0.95, max_level)
        )
        approxs.append(interpolation)
        if levels[i] in plot_points:
            ax[0].plot(
                process.saving_rates,
                n / np.trapz(n, x=process.saving_rates),
                label=f"M={process.number_of_levels}",
                color=cmap(i),
            )
        ax[0].legend()
        ax[0].set_yscale("log")
        ax[0].grid(True)
    # increase_in_precisions = [np.linalg.norm(approxs[i]-approxs[i+1]) for i in range(len(approxs)-1)]
    step_distances = [
        wasserstein_distance(
            u_values=processes[i - 1].saving_rates,
            v_values=processes[i].saving_rates,
            u_weights=processes[i - 1].unpacking_occupation_numbers()[-1],
            v_weights=processes[i].unpacking_occupation_numbers()[-1],
        )
        for i in np.arange(1, len(processes))
    ]
    final_distance = [
        wasserstein_distance(
            u_values=processes[i].saving_rates,
            v_values=processes[-1].saving_rates,
            u_weights=processes[i].unpacking_occupation_numbers()[-1],
            v_weights=processes[-1].unpacking_occupation_numbers()[-1],
        )
        for i in range(len(approxs))
    ]
    ax[1].plot(
        levels[1:-1],
        step_distances[:-1],
        color="red",
        label="dist to prvious distribution",
    )

    ax[1].plot(
        levels[1:-1],
        final_distance[1:-1],
        color="green",
        label="dist to final distribution",
    )
    for i, process in enumerate(processes):
        if levels[i] in plot_points:
            ax[1].scatter(
                [levels[i]],
                [final_distance[i]],
                color=cmap(i),
                zorder=5,
                edgecolors="k",
                s=50,
            )
            ax[1].scatter(
                [levels[i]],
                [step_distances[i - 1]],
                color=cmap(i),
                zorder=5,
                edgecolors="k",
                s=50,
            )
    ax[1].set_yscale("log")
    ax[1].grid(True)
    plt.savefig("SM_Fig6.png")
    plt.close()
    ###################################################################################################################
    # Wasserstein direct
    ###################################################################################################################
    fig, ax = plt.subplots(2)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "custom",
        ["tab:blue", "tab:purple", "tab:red", "tab:orange", "gold"],
        len(processes),
    )
    approxs = []
    for i, process in tqdm(enumerate(processes)):
        n = process.unpacking_occupation_numbers()[-1]
        interpolation = np.interp(
            np.linspace(0.05, 0.95, max_level), process.saving_rates, n
        )
        interpolation = interpolation / np.trapz(
            interpolation, x=np.linspace(0.05, 0.95, max_level)
        )
        approxs.append(interpolation)
        if levels[i] in plot_points:
            ax[0].plot(
                process.saving_rates,
                n / np.trapz(n, x=process.saving_rates),
                label=f"M={process.number_of_levels}",
                color=cmap(i),
            )
        ax[0].legend(ncol=4)
        ax[0].set_yscale("log")
        ax[0].grid(True)
    # increase_in_precisions = [np.linalg.norm(approxs[i]-approxs[i+1]) for i in range(len(approxs)-1)]
    step_distances = [
        wasserstein_distance(
            u_values=processes[i - 1].saving_rates,
            v_values=processes[i].saving_rates,
            u_weights=processes[i - 1].unpacking_occupation_numbers()[-1],
            v_weights=processes[i].unpacking_occupation_numbers()[-1],
        )
        for i in np.arange(1, len(processes))
    ]
    final_distance = [
        wasserstein_distance(
            u_values=processes[i].saving_rates,
            v_values=processes[-1].saving_rates,
            u_weights=processes[i].unpacking_occupation_numbers()[-1],
            v_weights=processes[-1].unpacking_occupation_numbers()[-1],
        )
        for i in range(len(approxs))
    ]
    ax[1].plot(
        levels[1:-1],
        step_distances[:-1],
        color="red",
        label="dist to prvious distribution",
    )
    for i, process in enumerate(processes):
        if levels[i] in plot_points:
            ax[1].scatter(
                [levels[i]],
                [step_distances[i - 1]],
                color=cmap(i),
                zorder=5,
                edgecolors="k",
                s=50,
            )
    ax[1].set_yscale("log")
    ax[1].grid(True)
    plt.savefig("convergence_final_step_dist.svg")
    plt.close()
    np.save(f"distance_between_steps.npy", np.array([levels[1:], step_distances]))
    np.save(f"distance_to_last_step.npy", np.array([levels, final_distance]))

    fig, ax = plt.subplots(2)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "custom",
        ["tab:blue", "tab:purple", "tab:red", "tab:orange", "gold"],
        len(processes),
    )
    approxs = []
    for i, process in tqdm(enumerate(processes)):
        n = process.unpacking_occupation_numbers()[-1]
        interpolation = np.interp(
            np.linspace(0.05, 0.95, max_level), process.saving_rates, n
        )
        interpolation = interpolation / np.trapz(
            interpolation, x=np.linspace(0.05, 0.95, max_level)
        )
        approxs.append(interpolation)
        if levels[i] in plot_points:
            ax[0].plot(
                process.saving_rates,
                n / np.trapz(n, x=process.saving_rates),
                label=f"M={process.number_of_levels}",
                color=cmap(i),
            )
        ax[0].legend(ncol=5)
        ax[0].set_yscale("log")
        ax[0].grid(True)
    # increase_in_precisions = [np.linalg.norm(approxs[i]-approxs[i+1]) for i in range(len(approxs)-1)]
    step_distances = [
        wasserstein_distance(
            u_values=processes[i - 1].saving_rates,
            v_values=processes[i].saving_rates,
            u_weights=processes[i - 1].unpacking_occupation_numbers()[-1],
            v_weights=processes[i].unpacking_occupation_numbers()[-1],
        )
        for i in np.arange(1, len(processes))
    ]
    final_distance = [
        wasserstein_distance(
            u_values=processes[i].saving_rates,
            v_values=processes[-1].saving_rates,
            u_weights=processes[i].unpacking_occupation_numbers()[-1],
            v_weights=processes[-1].unpacking_occupation_numbers()[-1],
        )
        for i in range(len(approxs))
    ]
    ax[1].plot(
        levels[1:-1],
        step_distances[:-1],
        color="red",
        label="dist to prvious distribution",
    )
    for i, process in enumerate(processes):
        if levels[i] in plot_points:
            ax[1].scatter(
                [levels[i]],
                [step_distances[i - 1]],
                color=cmap(i),
                zorder=5,
                edgecolors="k",
                s=50,
            )
    ax[1].set_yscale("log")
    ax[1].grid(True)
    plt.savefig("SM_Fig6_2.svg")
    plt.close()
