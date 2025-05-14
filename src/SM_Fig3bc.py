"""
Functions:
    - plot_SM_Fig3a(): 
        Loads data for beta = 50 from multiple runs and generates Figure 3a.

    - plot_SM_Fig3b(): 
        Loads data for beta = 15 from multiple runs and generates Figure 3b.

Data Format:
    Each data file is stored as a CSV-like text file with the following structure:
        - The first row contains the time array `t`.
        - Each subsequent row contains occupation numbers for different saving rate levels.
        - The shape of the stored data is (number_of_levels + 1, len(t)), 
          where the first row corresponds to `t` and the remaining rows correspond to the levels.

File Naming Convention:
    Data files are located in "data/SM_Fig3/" and follow the format:
        "occupation_number_time_series_beta={beta}_run{i}.txt"

NOTE: The exasct runs used to generate teh data can be requested from the authors!
But are not stored with the repository.

"""
from sigmoidal_transition_prob import *
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker

def find_spike_times(solution):
    """
    Identify the times at which the system transitions between the two most populated levels.

    The function detects sign changes in the difference between the occupation numbers 
    of the two dominant saving rate levels (n_0 and n_1), which characterize the switching behavior.
    It uses linear interpolation to estimate the exact transition times.

    Parameters:
    -----------
    solution : - Economic_Process

    Returns:
    --------
    list of float
        A list of estimated times at which transitions (spikes) occur.
    """
    if isinstance(solution, Economic_Process_sigmoidal):
        n = np.transpose(solution.unpacking_occupation_numbers())
        t = solution.times
    else:
        t = solution[0]
        n = solution[1:]

    # we want to find the transitions, which are characterized by
    # the n_2 = n_1, since the majority switches between these two levels
    y = n[0] -n[1]
    zeros = []
    for i in range(len(y) - 1):
        if y[i] * y[i + 1] < 0:  # Sign change
            # linear interpolation
            x_zero = t[i] - y[i] * (t[i+1] - t[i]) / (y[i+1] - y[i])
            zeros.append(x_zero)
    return zeros

def generate_data(filename):
    """
    Generates economic process data.

    This function initializes an Economic_Process_sigmoidal instance with 
    predefined parameters, solves the associated stochastic differential equation (SDE),
    and stores the resulting occupation number time series data in a text file.

    Parameters:
        filename (str): 
            The name of the output file (excluding the directory path and extension).

    Process Details:
        - The simulation runs for T = 100,000 time steps.
        - The system has 5 saving rate levels.
        - Agents are initially distributed equally across levels.
        - The capital distribution is initialized with mean = 100 and variance = 30.
        - The process uses an exploration rate of 0.05.

    Data Format:
        - The saved file is stored in "data/SM_Fig3/{filename}.txt".
        - The first row contains the time array `t`.
        - The subsequent rows contain occupation numbers for each saving rate level.
        - The final shape of the stored data is (num_levels + 1, len(t)).

    Returns:
        None
    """
    T=100_000
    tau = 300
    mean = 100
    var = 30
    sdev = np.sqrt(var)
    skewness = 0
    third_mom = sdev**3 * skewness + 3 * mean * sdev**2 + mean**3
    num_levels = 5
    initial_moments = [
            [mean for _ in range(5)],
            [var for _ in range(5)],
            [third_mom for _ in range(5)],
        ]
    process = Economic_Process_sigmoidal(
        number_of_agents=num_of_agents,
        number_of_levels=5,
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

    process.solve_SDE_economics(T,1)
    n = np.transpose(process.unpacking_occupation_numbers())
    t = process.times
    combined = np.vstack([t, n])
    np.savetxt(f"data/SM_Fig3/{filename}.txt", combined)

def main(files):
    """
    Analyze and visualize resting times between spikes in occupation numbers.

    This function loads multiple simulation runs of an economic process, extracts spike 
    times from occupation number trajectories, and categorizes the time intervals between 
    successive spikes into high-saving and low-saving states. It then plots a histogram 
    of these resting times with probability density estimates.

    Returns:
    --------
    None
        The function produces a histogram plot and prints the number of observed intervals
        for each saving state.
    """
    spike_times = []
    for file in files:
            process = np.loadtxt(file)
            spike_times.append(find_spike_times(process))

    high_saving = []
    low_saving = []
    for _, t_times in enumerate(spike_times):
        for i, t in enumerate(t_times):
            if i > 0 and i < len(t_times) - 1:
                delta_t = t - t_times[i - 1]
                if i % 2 == 1:
                    high_saving.append(delta_t)
                else:
                    low_saving.append(delta_t)

    # the histogram of the data
    fig, ax = plt.subplots(1, 1)
    ax.hist(
        low_saving,
        bins=30,
        density=True,
        facecolor="k",
        alpha=1,
        label=r"$\langle s_i \rangle = 0.3 $"
        + "\n"
        + r"$\mathbb{E}(T) = $"
        + str(int(np.round(np.mean(low_saving)))),
    )
    ax.hist(
        high_saving,
        bins=30,
        density=True,
        facecolor="red",
        alpha=0.7,
        label=r"$\langle s_i \rangle = 0.11 $"
        + "\n"
        + r"$\mathbb{E}(T) = $"
        + str(int(np.round(np.mean(high_saving)))),
    )
    print(
        f"We observed {len(low_saving)} intervals for the low saving "
        f"and {len(high_saving)} intervals for the high savin state"
    )

    plt.title(
        f"mean resting time s=0.05:{np.round(np.mean(low_saving))}, mean resting time s=0.24:{np.round(np.mean(high_saving))}"
    )
    ax.set_xlabel("Retsing Times T")
    ax.set_ylabel(r"Probability Distribution $p(T)$")
    ax.grid(True)
    ax.legend()
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax.yaxis.set_major_formatter(formatter)
    plt.show()

def plot_SM_Fig3b():
    """
    Plots Supplementary Material Figure 3b.

    Loads and processes occupation number time series data for beta = 15 
    from multiple simulation runs and generates the corresponding figure.

    The function retrieves data from the stored files in 
    "data/SM_Fig3/occupation_number_time_series_beta=15_run{i}.txt" 
    and passes them to the main plotting function.

    Returns:
        None
    """
    files = [f"data/SM_Fig3/occupation_number_time_series_beta=15_run{i}.txt" for i in range(17)]
    main(files)

def plot_SM_Fig3a():
    """
    Plots Supplementary Material Figure 3a.

    Loads and processes occupation number time series data for beta = 50 
    from multiple simulation runs and generates the corresponding figure.

    The function retrieves data from the stored files in 
    "data/SM_Fig3/occupation_number_time_series_beta=50_run{i}.txt" 
    and passes them to the main plotting function.

    Returns:
        None
    """
    files = [f"data/SM_Fig3/occupation_number_time_series_beta=50_run{i}.txt" for i in range(7)]
    main(files)

if __name__ == "__main__":
    plot_SM_Fig3a()
    plot_SM_Fig3b()
