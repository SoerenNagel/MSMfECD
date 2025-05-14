"""
This script is used to analyse the correlation time and produce Fig.4 in the paper.
There are two main functions: 
    - main(), which runs the simulations and calculates the correaltion time.
    - plot_from_paper(), which uses previously generated data from the files
        "correlation_time_beta=5.txt"
        "correlation_time_beta=8.txt"
        "correlation_time_beta=50.txt"
        Within these files, data stored in 4 rows: number of agents, noise intensity, average correlation time, standard error
Computing the full plots from scratch can be fairly expensive, since for for very large values of beta,
there is some probability, that the consumption becomes to large during a switch between the meta stable states.
This results in Nan's, which make the calculation invalid. As a consequence a large number of runs have to
be used, to calculate the correlation time.

Usage:

Each value of beta gets it's own data file. So for a diagram with two values of beta.

main(beta1, "file1.txt")
main(beta2, "file2.txt")
"""

from sigmoidal_transition_prob import Economic_Process_sigmoidal
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import tqdm
from matplotlib import ticker


class CorrelationTime:
    """
    This class calculated the correlation time for a single instance of the Economic Process.
    The correlation time should be averaged over as many runs as possible anf for a long integration 
    time. Most parameters are identical to the EconomicProcess class and wil be passed over.

    param: cut_off_time
        The number of time steps that will be removed from the timeseries of the process.
    param: number_of_runs
        Nuber of runs for which the correlation time will be averaged.


    """
    def __init__(
        self,
        number_of_agents: int,
        number_of_levels: int,
        update_time: float,
        temperature: float,
        initial_occupation_numbers,
        exploration: float,
        initial_capital_moments: list,
        labour: float,
        elasticities: list,
        production_constant: float,
        depreciation: float,
        target_time: int,
        solver_rate:int,
        cut_off_time: int,
        number_of_runs: int,
        gamma: float,
    )->None:
        self.number_of_agents = number_of_agents
        self.number_of_levels = (
            number_of_levels  # needs to be generated from initial data
        )
        self.update_time = update_time
        self.temperature = temperature
        self.initial_occupation_numbers = np.array(
            initial_occupation_numbers, dtype=np.float64
        )
        self.exploration = exploration

        # defining data for the market process
        self.labour = labour
        self.elasticities = np.array(elasticities)
        self.production_constant = production_constant
        self.depreciation = depreciation
        # here we are defining functions for the market process
        self.initial_capital_moments = np.array(
            initial_capital_moments, dtype=np.float64
        )
        # use data structure, s.t. capital_moments = [[first moments], [second moments], [third moments]]
        self.target_time = target_time
        self.solver_rate = solver_rate
        self.cut_off_time = cut_off_time
        self.number_of_runs = number_of_runs
        self.gamma = gamma
        self.correlation_time: None| np.ndarray = None
    def init_and_solve_process(self,_)->float:
        """
        This functions runs the simulation and calculated the correlation time.
        It will mainly be passed to multiprocessing in calculate_correlation_time().
        """
        process = Economic_Process_sigmoidal(
            number_of_agents=self.number_of_agents,
            number_of_levels=self.number_of_levels,
            update_time=self.update_time,
            temperature=self.temperature,
            initial_occupation_numbers=self.initial_occupation_numbers,
            initial_capital_moments=self.initial_capital_moments,
            labour=self.number_of_agents,
            elasticities=self.elasticities,
            production_constant=self.production_constant,
            depreciation=self.depreciation,
            exploration=self.exploration,
            gamma=self.gamma,
        )

        process.solve_SDE_economics(self.target_time, self.solver_rate)
        corr_time = process.correlation_time(self.cut_off_time)
        return corr_time

    def calculate_correlation_times(self)->None:
        """
        Simulate independat runs and caluclate the correlation times.The 
        reults will be stored in self.correlation_time.
        """
        with mp.Pool(mp.cpu_count()) as pool:
            correlation_time = list(pool.imap_unordered(self.init_and_solve_process, range(self.number_of_runs)))
            correlation_time = [tau for tau in correlation_time if not np.isnan(tau)]
        self.correlation_time = np.array(correlation_time)

    def get_data(self):
        """
        Get the relevant data for the visualization. Both number of agents and noise intensity,
        aswell as the averaged correlation time, with the standard error, since the runs are
        uncorrelated.

        return:
            N, Gama, tau_c, Standard Error of tau_c
        """
        assert self.correlation_time is not None, "Correlation time has not been claculated."
        data = np.zeros(4)
        data[0] = self.number_of_agents
        data[1] = self.gamma if self.labour != self.number_of_agents else 1/np.sqrt(self.number_of_agents)
        data[2] = np.mean(self.correlation_time)
        data[3] = np.sqrt(np.var(self.correlation_time) / len(self.correlation_time))
        return data



def plot_from_paper() -> None:
    """
    Uses previously generated data to combine correlation time Plots for several paramerters.
    """
    cm = 1 / 2.54
    _, axs = plt.subplots(figsize=(8.1944 * cm, 8 * cm))
    ax = axs.twiny()
    data_beta_5 = np.loadtxt("data/correlation_time/correlation_time_beta=5.txt")
    data_beta_8 = np.loadtxt("data/correlation_time/correlation_time_beta=8.txt")
    data_beta_50 = np.loadtxt("data/correlation_time/correlation_time_beta=50.txt")
    new_markers = ["s", "p", "d"]
    new_colors = ["tab:blue", "tab:orange", "tab:red"]
    new_beta = [5, 8, 50]
    for i, data in enumerate([data_beta_5, data_beta_8, data_beta_50]):
        number_of_agents, noise_inetensity, average_correlation_time, _ = data
        axs.scatter(
            noise_inetensity,
            average_correlation_time,
            marker=new_markers[i],
            label=r"$\beta=$" + f"{new_beta[i]}",
            s=10,
            color=new_colors[i],
        )
        ax.plot(
            number_of_agents, average_correlation_time, linewidth=1, color=new_colors[i]
        )

    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    axs.yaxis.set_major_formatter(formatter)
    axs.set_xlabel(r" Noise Intensity $\Gamma$")
    ax.set_xlabel(r"Number of Agents $N$")
    axs.set_ylabel(r"$\tau_c$")
    scale = lambda x: 1 / np.sqrt(x)
    inv_scale = lambda x: 1 / x**2
    ax.set_xscale("function", functions=(scale, inv_scale))
    axs.legend()
    axs.grid()
    plt.show()


def main(beta:float, filename:str):
    """
    This is the main function for data generation. 
    It will generate the data for a single plot like Fig.4 in the paper,
    with a fixed value of beta.

    The data will be stored in the file, wich should have the form "example.txt"

    To generate the full Fig from scratch, run this for all relevant values of beta.
    This can be somewhat expensive. The data used for the paper is stored in the repo
    and can be plotted using: plot_from_paper()
    """
    number_of_runs = 5
    target_time = 50_00
    tau = 300
    mean = 100
    var = 30
    sdev = np.sqrt(var)

    skewness = 0
    third_mom = sdev**3 * skewness + 3 * mean * sdev**2 + mean**3
    num_levels = 5
    initial_moments = [
        [mean for _ in range(num_levels)],
        [var for _ in range(num_levels)],
        [third_mom for _ in range(num_levels)],
    ]
    min_agent = 150
    max_agent = 1000
    steps = 8

    agents = np.linspace(min_agent, max_agent, steps)
    data = np.zeros((steps, 4))

    # The main diagram axis is supposed to be the noise intensity which is
    # given by 1/sqrt(N), so we order it by ascending by noise intensity.
    for i, N in tqdm.tqdm(enumerate(reversed(agents))):
        c_time = CorrelationTime(
            number_of_agents=N,
            number_of_levels=num_levels,
            update_time=tau,
            temperature=1 / beta,
            initial_occupation_numbers=np.full(num_levels, N / num_levels),
            initial_capital_moments=initial_moments,
            labour=N,
            elasticities=[0.5, 0.5],
            production_constant=1,
            depreciation=0.05,
            exploration=0.05,
            target_time=target_time,
            solver_rate=1,
            cut_off_time=2000,
            number_of_runs=number_of_runs,
            gamma=1,
        )
        c_time.calculate_correlation_times()
        data[i] = c_time.get_data()
    np.savetxt(filename, data)


if __name__ == "__main__":
    plot_from_paper()
    # main(beta=8, filename="test.txt")
