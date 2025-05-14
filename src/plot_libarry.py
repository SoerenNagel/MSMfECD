import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import phase_space_functions as psf
from itertools import product
import matplotlib.colors as colors
from mpl_point_clicker import clicker
from mpl_interactions import zoom_factory, panhandler
from matplotlib import ticker

def plot_skewness_illustration_5_level(solution) -> None:
    levels = [0, 1, 4]
    t = solution.times
    moments = np.array(solution.capital_moments)
    occupation_numbers = solution.unpacking_occupation_numbers()
    con_mom = np.array([psf.moments_of_consumption_distribution(moments[i][0], moments[i][1], moments[i][2],
                                                       occupation_numbers[i],
                                                       solution.production_constant, solution.elasticities,
                                                       solution.labour, solution.number_of_levels,
                                                       solution.saving_rates, solution.number_of_agents) for i in
               range(len(t))])
    skewness = np.array([psf.skewness(c_mom) for c_mom in con_mom])
    skewness = np.transpose(skewness)
    cap_rent = [psf.return_on_investment(moments[i][0], solution.production_constant, solution.elasticities, solution.labour, occupation_numbers[i]) for i, _ in enumerate(t)]
    mean_saving_rates = [np.dot(solution.saving_rates, occupation_numbers[i]/solution.number_of_agents) for i, _ in enumerate(t)]

    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    fig.subplots_adjust(top=.85)

    ax[0, 0].set_title(r'$s=$' + str(np.around(solution.saving_rates[1], 2)))
    ax[0, 0].set_xlabel(r'$t$')
    ax[0, 0].set_ylabel(r'$\langle s \rangle$')
    ax[0, 0].yaxis.set_major_formatter(formatter)
    lns1 = ax[0, 0].plot(t[1500:], mean_saving_rates[1500:], color='tab:orange', alpha=0.7, label=r'$\langle s \rangle$')
    axs = ax[0, 0].twinx()
    axs.yaxis.set_major_formatter(formatter)
    lns2 = axs.plot(t[1500:], con_mom[1500:, 0, levels[1]], label=r'$\langle C_l\rangle$')
    axs.set_ylabel(r'$\langle C\rangle$')
    axs.grid(True)
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax[0, 0].legend(lns, labs, loc=0)

    ax[0, 1].set_title(r'$s=$' + str(np.around(solution.saving_rates[-1], 2)))
    ax[0, 1].set_xlabel(r'$t$')
    ax[0, 1].set_ylabel(r'$\langle s \rangle$')
    ax[0, 1].yaxis.set_major_formatter(formatter)
    lns1 = ax[0, 1].plot(t[1500:], mean_saving_rates[1500:], color='tab:orange', alpha=0.7, label=r'$\langle s \rangle$')
    axs = ax[0, 1].twinx()
    axs.yaxis.set_major_formatter(formatter)
    lns2 = axs.plot(t[1500:], con_mom[1500:, 0, levels[2]], label=r'$\langle C\rangle$')
    axs.set_ylabel(r'Skew$(C)$')
    axs.grid(True)
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax[0, 1].legend(lns, labs, loc=0)


    ax[1, 0].set_title(r'$s=$' + str(np.around(solution.saving_rates[1], 2)))
    ax[1, 0].set_xlabel(r'$t$')
    ax[1, 0].set_ylabel(r'$\langle s \rangle$')
    ax[1, 0].yaxis.set_major_formatter(formatter)
    lns1 = ax[1, 0].plot(t[1500:], cap_rent[1500:], color='tab:red', alpha=0.7,
                         label=r'r')
    axs = ax[1, 0].twinx()
    axs.yaxis.set_major_formatter(formatter)
    axs.set_ylabel(r'$\langle C \rangle')
    lns2 = axs.plot(t[1500:], con_mom[1500:,0, levels[1]], label=r'$\langle C \rangle$')
    axs.set_ylabel(r'$\langle C \rangle$')
    axs.grid(True)
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax[1, 0].legend(lns, labs, loc=0)

    ax[1, 1].set_title(r'$s=$' + str(np.around(solution.saving_rates[-1], 2)))
    ax[1, 1].set_xlabel(r'$t$')
    ax[1, 1].set_ylabel(r'$\langle s \rangle$')
    ax[1, 1].yaxis.set_major_formatter(formatter)
    lns1 = ax[1, 1].plot(t[1500:], cap_rent[1500:], color='tab:red', alpha=0.7,
                         label=r'r')
    axs = ax[1, 1].twinx()
    axs.yaxis.set_major_formatter(formatter)
    axs.set_ylabel(r'$\langle C \rangle')
    lns2 = axs.plot(t[1500:], con_mom[1500:,0, levels[2]], label=r'$\langle C \rangle$')
    axs.set_ylabel(r'$\langle C \rangle$')
    axs.grid(True)
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax[1, 1].legend(lns, labs, loc=0)

    fig.subplots_adjust(top=.85)
    plt.show()

def plot_single_time_series_occupation_numbers(
    solution, record_coordinates=False
) -> None:
    """
    Plot the time series of occupation numbers for different saving rate levels.

    This function visualizes how the number of agents at each saving rate level evolves 
    over time. The y-axis is set to a symmetric logarithmic scale to accommodate 
    variations in occupation numbers. Optionally, it allows recording user-selected 
    transition points via mouse clicks.

    Parameters:
    -----------
    solution : Economic_Process
        A solved instance of the economic process containing occupation number data.
    record_coordinates : bool, optional (default=False)
        If True, enables interactive tools for zooming, panning, and clicking to record 
        transition points. The x-coordinates of clicked points are returned.

    Returns:
    --------
    ndarray or None
        If `record_coordinates` is True, returns an array of recorded x-coordinates 
        corresponding to transition points. Otherwise, returns None.
    """
    t = solution.times
    n = np.transpose(solution.unpacking_occupation_numbers())
    fig, axs = plt.subplots()
    axs.set_title("Occupation Numbers n")
    axs.set_xlabel("$t$")
    for level in range(solution.number_of_levels):
        axs.plot(
            t, n[level], label=r"$s=$" + str(np.round(solution.saving_rates[level], 2))
        )
    axs.grid(True)
    axs.set_yscale("symlog")
    axs.legend(loc="upper left")
    if record_coordinates == True:
        zoom_factory(axs)
        ph = panhandler(fig, button=2)
        klicker = clicker(axs, ["transitions"], markers=["x"])
    plt.show()
    if record_coordinates == True:
        return klicker.get_positions()["transitions"][:, 0]


def plot_simple_time_series(
    solution, plot_mean_saving_rate=True, production_in_percentage=False
) -> None:
    t = solution.times
    n = solution.unpacking_occupation_numbers()
    s = solution.saving_rates
    moments = np.array(solution.capital_moments)
    means = np.array([moments[i][0] for i in range(len(moments))])
    production = np.array(
        [
            psf.production(
                means[i],
                solution.production_constant,
                n[i],
                solution.elasticities,
                solution.labour,
            )
            for i in range(len(t))
        ]
    )
    if production_in_percentage:
        production = (production - np.mean(production)) / np.mean(production)

    fig, ax = plt.subplots(3, 1, sharex=True)
    ax[0].grid(True)
    ax[1].grid(True)
    ax[2].grid(True)
    ax[0].set_yscale("symlog")
    ax[1].set_yscale("symlog", linthresh=300)
    ax[0].set_ylabel("$n(t)$")
    ax[1].set_ylabel(r"$\langle K_i \rangle _l$")
    ax[0].set_title("Occupation Numbers")
    ax[1].set_title("Capital Means")
    ax[2].set_title("Economic Production")
    fig.tight_layout()
    pars = [
        solution.number_of_agents,
        solution.update_time,
        solution.labour,
        solution.production_constant,
        solution.depreciation,
        solution.elasticities[0],
        solution.elasticities[1],
        solution.relative_noise,
        1 / solution.temperature,
    ]
    [N, tau, L, _, delta, alpha, beta, Noise, inv_temp] = [str(par) for par in pars]
    fig.suptitle(
        r"Parameters: $N = $"
        + N
        + r", $\tau = $"
        + tau
        + ", $L=$"
        + L
        + r", $\delta=$"
        + delta
        + r", $\alpha = $"
        + alpha
        + r", $\beta_{elasticity}$ ="
        + beta
        + r"relative Noise = "
        + str(Noise)
        + r" $\beta_{T}=$ "
        + str(inv_temp)
    )

    for level in range(solution.number_of_levels):
        # color = colors[level]
        ax[0].plot(
            t, np.transpose(n)[level], label="$s =" + str(np.around(s[level], 2)) + "$"
        )
        ax[1].plot(t, np.transpose(means)[level])
    ax[2].plot(t, production, label=r"Production Y", color="k")
    if plot_mean_saving_rate == True:

        av_saving_rate = np.zeros(len(n))
        for i, nums in enumerate(n):
            av_saving_rate[i] = (
                np.dot(nums, solution.saving_rates) / solution.number_of_agents
            )
        axs = ax[2].twinx()
        axs.plot(
            t, av_saving_rate, label=r" Average saving saving rate $\langle s \rangle$"
        )
        axs.legend()
    ax[0].legend(
        bbox_to_anchor=(0.5, -0.27), loc="lower center", ncol=10, handlelength=1
    )
    ax[2].legend()
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax[2].yaxis.set_major_formatter(formatter)
    fig.subplots_adjust(top=0.85)
    plt.show()


def plot_production_and_mean_saving_rate(solution) -> None:
    """
    Plots the production and mean saving rate for the solution. 
    This is the Function that is used for Fig.5 in the Supplementary material.
    """
    _, axs = plt.subplots()
    t = solution.times
    #########################################
    # occupation_number_data
    n = solution.unpacking_occupation_numbers()
    n0=np.sum(n[0])
    n *=n0
    #########################################
    # capital distribution data
    moments = np.array(solution.capital_moments)
    ###########################################
    # data for economic production
    prod = np.array(
        [
            psf.production(
                moments[i][0],
                solution.production_constant,
                n[i],
                solution.elasticities,
                solution.labour*n0,
            )
            for i in range(len(t))
        ]
    )
    axs.plot(t, prod, label=r"$Y_t$", linewidth=0.8, color="black")
    axs.set_ylabel(r"Economic Production $Y$")
    axs.set_yscale("log")
    axs.set_xlabel("$t$")

    axs2 = axs.twinx()
    axs2.set_ylabel(r"Returns $\langle S_i \rangle $")

    av_saving_rate = np.zeros(len(n))
    for i, nums in enumerate(n):
        av_saving_rate[i] = (
            np.dot(nums, solution.saving_rates) / np.sum(nums)
        )
    axs2.plot(t, av_saving_rate, label=r"$\langle S_i \rangle $", color="tab:blue")

    axs.grid(True)
    plt.legend()
    plt.show()


def plot_occupation_numbers_over_capital_mean(solution, mode: str) -> None:
    n = solution.unpacking_occupation_numbers()
    means = np.array(
        [solution.capital_moments[i][0] for i in range(len(solution.times))]
    )
    if mode == "line":
        fig, axs = plt.subplots()
        # axs.set_title("Power Spectrum for single capital in the mean field")
        axs.set_ylabel("$n_l(t)$")
        axs.set_xlabel(r"$\langle K \rangle_l(t)$")
        for level in range(solution.number_of_levels):
            axs.plot(
                np.transpose(means)[level],
                np.transpose(n)[level],
                label=r"$s_l=$" + str(np.around(solution.saving_rates[level], 2)),
            )
        axs.grid(True)
        axs.set_yscale("symlog")
        axs.set_xscale("symlog")
        axs.legend()
        plt.show()
    elif mode == "heatmap":
        x = np.transpose(means)[0]
        y = np.transpose(n)[0]
        for level in range(solution.number_of_levels):
            if level > 0:
                x = np.concatenate((x, np.transpose(means)[level]))
                y = np.concatenate((y, np.transpose(n)[level]))
        x_range = np.linspace(np.amin(x), np.amax(x), 400)
        y_range = np.linspace(np.amin(y), np.amax(y), 400)

        x = np.log(np.array(x))
        y = np.log(np.array(y))
        heatmap, _, _ = np.histogram2d(x, y, bins=400)

        #######################################################################################################
        ##############################################################################################
        fig, ax = plt.subplots()
        ax.set_facecolor("black")
        print(np.sum(heatmap))
        cax = ax.pcolor(
            x_range,
            y_range,
            np.transpose(heatmap),
            norm=colors.LogNorm(vmin=0.01, vmax=heatmap.max()),
        )
        # cax = ax.pcolor(heatmap, norm=colors.SymLogNorm(linthresh=0.001, linscale=0.001,
        #                                      vmin=0, vmax=np.amax(heatmap), base=10))
        fig.colorbar(cax)
        fig.tight_layout()
        plt.show()


def plot_saving_rate_color_plot(solution) -> None:
    """
    """
    n = np.transpose(solution.unpacking_occupation_numbers())
    n*=np.sum(n[:,0])
    _, ax = plt.subplots()
    # ax.pcolormesh(np.log(n))
    im = ax.imshow(n, aspect="auto", interpolation="none", norm=colors.LogNorm())

    # Show all ticks and label them with the respective list entries
    # ax.set_xticks(np.arange(len(farmers)), labels=farmers)

    # Rotate the tick labels and set their alignment.
    # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
    #         rotation_mode="anchor")

    plt.colorbar(im)

    plt.show()


def plot_final_occupation_numbers_over_savingrate(solutions: list) -> None:
    _, axs = plt.subplots()
    for solution in solutions:
        n = solution.unpacking_occupation_numbers()
        axs.plot(
            solution.saving_rates,
            n[-1],
            label=r"$\beta =$ " + str(np.around(1 / solution.temperature)),
        )
    axs.set_title("Distribution of Saving Rates")
    axs.set_xlabel(r"saving rates $s_l$")
    axs.set_ylabel(r"occupation numbers $n_l$")
    axs.set_yscale("log")
    axs.legend()
    axs.grid()

    plt.show()


def plot_saving_rate_distribution(solution, times: list|np.ndarray) -> None:
    """
    This is the function, that plots the saving rate distribution in Fig1a
    """
    _, axs = plt.subplots()
    width = 1 / (len(times) * len(solution.saving_rates))
    for i, time in enumerate(times):
        n = solution.unpacking_occupation_numbers()[int(time)]
        print(solution.capital_moments[time])
        # n = solution.number_of_agents * n

        axs.bar(
            solution.saving_rates + i * width * np.ones(len(solution.saving_rates)),
            n,
            width=width,
            label=r"$\langle S_i \rangle$ = "
            + str(
                np.round(
                    np.dot(n, solution.saving_rates) / solution.number_of_agents, 2
                )
            ),
        )
        # axs.plot(solution.saving_rates, n[int(time)], label=f"t = {time}")
    axs.set_title(f"Distribution of Saving Rates at")
    axs.set_xlabel(r"saving rates $s_l$")
    axs.set_ylabel(r"occupation numbers $n_l$")
    axs.set_yscale("log")
    axs.legend()
    axs.grid()

    plt.show()


def plot_saving_rate_distribution_from_procces(solutions: list) -> None:
    """
    This is the function, that plots the saving rate distribution in Fig1a
    """
    _, axs = plt.subplots()
    width = 1 / (2 * len(solutions[0].saving_rates))
    for i, solution in enumerate(solutions):
        n = solution.unpacking_occupation_numbers()[-1]
        axs.bar(
            solution.saving_rates + i * width * np.ones(len(solution.saving_rates)),
            n,
            width=width,
            label=r"$\langle S_i \rangle$ = "
            + str(
                np.round(
                    np.dot(n, solution.saving_rates) / solution.number_of_agents, 2
                )
            ),
        )
    axs.set_title(f"Distribution of Saving Rates at")
    axs.set_xlabel(r"saving rates $s_l$")
    axs.set_ylabel(r"occupation numbers $n_l$")
    axs.set_yscale("log")
    axs.legend()
    axs.grid()

    plt.show()

def plot_SM_fig8cd(solutions: list, time: int = -1) -> None:
    _, axs = plt.subplots(1, 2, figsize=(6.5, 3))
    ns = []
    for solution in solutions:
        n = solution.unpacking_occupation_numbers()[int(time)]
        # n = solution.number_of_agents * n
        ns.append(n)
        axs[0].plot(solution.saving_rates, n)
        axs[1].plot(
            solution.saving_rates,
            solution.saving_rates * solution.capital_moments[time][0],
            label=r"$\epsilon=$" + str(np.round(solution.exploration, 2)),
        )

    axs[0].set_xlabel(r"$s_l$")
    axs[0].set_ylabel(r"$n_l$")
    axs[0].set_yscale("symlog", linthresh=8 * 10**3)

    axs[0].grid()

    axs[1].set_xlabel(r"$s_l$")
    axs[1].set_ylabel(r"$K_l $")
    axs[1].set_yscale("log")
    axs[1].legend(loc="upper left", fontsize="x-small", ncol=2)
    axs[1].grid()

    lax = axs[0].inset_axes([0.5, 0.62, 0.485, 0.37])  # subregion of the original image
    lax.plot(
        [sol.exploration for sol in solutions],
        [np.dot(n / np.sum(n), sol.saving_rates) for n, sol in zip(ns, solutions)],
        marker="x",
        color="k",
    )
    lax.set_xlabel(r"$\epsilon$")
    lax.set_ylabel(r"$\langle S_i \rangle$")
    lax.grid()

    rax = axs[1].inset_axes(
        [0.5, 0.085, 0.485, 0.37]
    )  # subregion of the original image
    rax.plot(
        [sol.exploration for sol in solutions],
        [np.dot(n, sol.capital_moments[time][0]) for n, sol in zip(ns, solutions)],
        marker="x",
        color="k",
    )
    rax.set_xlabel(r"$\epsilon$")
    rax.set_ylabel(r"$K$")
    rax.grid()

    plt.savefig("parameter_variation_epsilon.svg")
    plt.tight_layout()
    plt.show()

def plot_SM_fig8ab(solutions: list, time: int = -1) -> None:
    _, axs = plt.subplots(1, 2, figsize=(6.5, 3))
    ns = []
    for solution in solutions:
        n = solution.unpacking_occupation_numbers()[int(time)]
        # n = solution.number_of_agents * n
        ns.append(n)
        axs[0].plot(solution.saving_rates, n)
        axs[1].plot(
            solution.saving_rates,
            solution.saving_rates * solution.capital_moments[time][0],
            label=r"$\beta=$" + str(np.round(1/solution.temperature, 2)),
        )

    axs[0].set_xlabel(r"$s_l$")
    axs[0].set_ylabel(r"$n_l$")
    axs[0].set_yscale("symlog", linthresh=8 * 10**3)

    axs[0].grid()

    axs[1].set_xlabel(r"$s_l$")
    axs[1].set_ylabel(r"$K_l $")
    axs[1].set_yscale("log")
    axs[1].legend(loc="upper left", fontsize="x-small", ncol=2)
    axs[1].grid()

    lax = axs[0].inset_axes([0.5, 0.62, 0.485, 0.37])  # subregion of the original image
    lax.plot(
        [1/sol.temperature for sol in solutions],
        [np.dot(n / np.sum(n), sol.saving_rates) for n, sol in zip(ns, solutions)],
        marker="x",
        color="k",
    )
    lax.set_xlabel(r"$\beta$")
    lax.set_ylabel(r"$\langle S_i \rangle$")
    lax.grid()

    rax = axs[1].inset_axes(
        [0.5, 0.085, 0.485, 0.37]
    )  # subregion of the original image
    rax.plot(
        [1/sol.temperature for sol in solutions],
        [np.dot(n, sol.capital_moments[time][0]) for n, sol in zip(ns, solutions)],
        marker="x",
        color="k",
    )

    rax.set_xlabel(r"$\beta$")
    rax.set_ylabel(r"$K$")
    rax.grid()

    plt.savefig("parameter_variation_beta.svg")
    plt.tight_layout()
    plt.show()

def line_plot_saving_rate_distribution(solutions: list, time: int = -1) -> None:
    _, axs = plt.subplots(1, 2, figsize=(6.5, 3))
    ns = []
    for solution in solutions:
        n = solution.unpacking_occupation_numbers()[int(time)]
        n = solution.number_of_agents * n
        ns.append(n)
        axs[0].plot(solution.saving_rates, n)
        axs[1].plot(
            solution.saving_rates,
            solution.saving_rates * solution.capital_moments[time][0],
            label=r"$\epsilon=$" + str(np.round(solution.exploration, 2)),
        )

    axs[0].set_xlabel(r"$s_l$")
    axs[0].set_ylabel(r"$n_l$")
    axs[0].set_yscale("symlog", linthresh=8 * 10**3)

    axs[0].grid()

    axs[1].set_xlabel(r"$s_l$")
    axs[1].set_ylabel(r"$K_l $")
    axs[1].set_yscale("log")
    axs[1].legend(loc="upper left", fontsize="x-small", ncol=2)
    axs[1].grid()

    lax = axs[0].inset_axes([0.5, 0.62, 0.485, 0.37])  # subregion of the original image
    lax.plot(
        [sol.exploration for sol in solutions],
        [np.dot(n / np.sum(n), sol.saving_rates) for n, sol in zip(ns, solutions)],
        marker="x",
        color="k",
    )
    lax.set_xlabel(r"$\epsilon$")
    lax.set_ylabel(r"$\langle S_i \rangle$")
    lax.grid()

    rax = axs[1].inset_axes(
        [0.5, 0.085, 0.485, 0.37]
    )  # subregion of the original image
    rax.plot(
        [sol.exploration for sol in solutions],
        [np.dot(n, sol.capital_moments[time][0]) for n, sol in zip(ns, solutions)],
        marker="x",
        color="k",
    )
    rax.set_xlabel(r"$\epsilon$")
    rax.set_ylabel(r"$K$")
    rax.grid()

    plt.savefig("parameter_variation_epsilon.svg")
    plt.tight_layout()
    plt.show()


def plot_capital_distribution(solutions) -> None:
    _, axs = plt.subplots()
    for solution in solutions:
        n = solution.unpacking_occupation_numbers()
        means = solution.capital_moments[-1][0]
        axs.plot(
            means, n[-1], label=r"$\beta =$ " + str(np.around(1 / solution.temperature))
        )
    axs.set_title("Distribution of Saving Rates")
    axs.set_xlabel(r"mean capital $\langle K_i \rangle_l$")
    axs.set_ylabel(r"occupation numbers $n_l$")
    # axs.set_xscale('log')
    axs.set_yscale("log")
    axs.legend()
    axs.grid()

    plt.show()


def plot_transition_rates(solution) -> None:
    t = solution.times
    L = solution.number_of_levels

    #########################################
    # occupation_number_data
    n_bin = np.zeros((1, solution.number_of_levels))
    times_for_n = np.zeros(1)
    for sol in solution.social_process:
        n_bin = np.concatenate((n_bin, sol.occupation_numbers))
        times_for_n = np.concatenate((times_for_n, sol.time_points))
    n_bin = np.delete(n_bin, 0, axis=0)
    times_for_n = np.delete(times_for_n, 0)
    n_bin = np.transpose(n_bin)
    occupation_numbers = np.zeros((solution.number_of_levels, len(t)))
    for l in range(solution.number_of_levels):
        occupation_numbers[l] = np.interp(t, times_for_n, n_bin[l])
    n = np.transpose(occupation_numbers)
    ############################################
    moments = np.array(solution.capital_moments)
    t_rates = np.zeros(
        (len(t), L, L)
    )  # will get the structure t_tares[time][k][l] is the rate from level k to level l
    for i in range(len(t)):
        t_rates[i] = psf.transition_rates(
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
    ) * np.ones(len(t))
    for i in range(len(t)):
        for l in range(L):
            number_of_transitions[i] += -t_rates[i][l][l]
    cm = 1 / 2.54  # centimeters in inches
    _, axs = plt.subplots(2, 1, figsize=(8.81944 * cm, 6 * cm))

    # The mulitplication by number_of_agents in the line below fixes the wrong SDE
    axs[1].plot(
        t,
        solution.number_of_agents * number_of_transitions,
        label=r"$\frac{N}{\tau}-\sum_{l=1}^N \alpha_{s_l \rightarrow s_l}$",
        color="k",
        linewidth=1.4,
        linestyle="dashed",
    )
    # axs[1].set_title("Transition Rates " + r'$\alpha_{s_l \rightarrow s_k}$')
    # axs[1].set_ylabel(r'$\alpha_{s_l \rightarrow s_k}$')

    for k, l in product(range(L), range(L)):
        if not (k == l):
            rates = np.array([t_rates[i][k][l] for i in range(len(t))])
            # axs[1].plot(t, rates, label= str(np.around(solution.saving_rates[k], 2 )) + r'$\rightarrow$' + str(np.around(solution.saving_rates[l], 2)), linewidth=.8)
            if (
                k > l
                and not (k == 4 and l == 3)
                and not (k == 4 and l == 2)
                and not (k == 3 and l == 2)
            ):
                rev_rates = np.array([t_rates[i][l][k] for i in range(len(t))])
                # In the line below, I have corrected for the wrong SDE by multiplying with number of agents!!!
                axs[0].plot(
                    t,
                    solution.number_of_agents * (rates - rev_rates),
                    label=str(np.around(solution.saving_rates[k], 2))
                    + r"$\rightarrow$"
                    + str(np.around(solution.saving_rates[l], 2)),
                    linewidth=1.4,
                    alpha=0.8,
                )

    # axs[2].legend()
    axs[1].set_yscale("log")
    axs[0].set_yscale("symlog", linthresh=20)
    # axs[2].set_yscale('symlog', linthresh=.2)
    axs[1].set_xlabel("$t$")
    axs[0].legend()
    axs[1].legend()
    axs[0].grid(True)
    axs[1].grid(True)
    # axs[2].grid(True)
    plt.show()


def plot_bifurcation_diagramm(
    bifurcation_data,
    betas,
    process,
    critical_points=None,
    back_tracking=None,
) -> None:
    states = bifurcation_data[0]
    fixed_points_n = bifurcation_data[1]
    # fixed_points_moments = bifurcation_data[2]
    _, axs = plt.subplots()
    for state_num in range(len(states)):
        mean_saving_rate = np.zeros(len(betas))
        for j, _ in enumerate(betas):
            mean_saving_rate[j] = (
                np.dot(process.saving_rates, fixed_points_n[state_num][j])
                / process.number_of_agents
            )
        axs.plot(
            betas, mean_saving_rate, lw=2, label=f"state_num={state_num}", color="k"
        )
        if critical_points is not None:
            critical_points = np.array(critical_points)
            for i, index in enumerate(critical_points):
                print(f"first test iteration {i}: {not np.isnan(index)}, {state_num}")
                if (not np.isnan(index)) and i == state_num:
                    axs.scatter(
                        betas[int(index)],
                        mean_saving_rate[int(index)],
                        marker="D",
                        color="tab:red",
                    )
    if back_tracking is not None:
        bt_states = back_tracking[0]
        bt_fixed_point_n = back_tracking[1]
        # bt_fixed_point_mom = back_tracking[2]
        bt_betas = back_tracking[3]
        for state_num, _ in enumerate(bt_states):
            mean_saving_rate = np.zeros(len(bt_betas[state_num]))
            for j, _ in enumerate(bt_betas[state_num]):
                mean_saving_rate[j] = (
                    np.dot(process.saving_rates, bt_fixed_point_n[state_num][j])
                    / process.number_of_agents
                )
            axs.plot(
                bt_betas[state_num],
                mean_saving_rate,
                lw=2,
                label=f"state_num={state_num}",
                color="tab:gray",
            )
    axs.set_ylabel(r"Mean Saving Rate $\langle s_i \rangle$")
    axs.set_xlabel(r"$\beta$")
    axs.grid(True)
    plt.show()


def plot_analayse_states_from_bifurcation_diagram(solutions: list) -> None:
    """
    This function plots the distribution of saving rates and analyzes the effects 
    of the mean saving rate on economic production and wealth inequality.

    Parameters
    ----------
    solutions : list
        A list of solution objects. Each solution must be and instance of Economic Process Sigmoidal
    
    Returns
    -------
    None
        Displays two plots:
        1. The distribution of saving rates.
        2. The relationship between the mean saving rate, production, and wealth inequality.

    """
    solutions = sorted(
        solutions,
        key=lambda process: np.dot(
            process.saving_rates, process.initial_occupation_numbers
        )
        / process.number_of_agents,
    )
    _, axs = plt.subplots()
    
    # Set up the the custom colormap
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "custom",
        ["tab:blue", "tab:purple", "tab:red", "tab:orange", "gold"],
        len(solutions),
    )

    # plot the saving rate distributions for each solution
    for i, solution in enumerate(solutions):
        n = solution.initial_occupation_numbers
        axs.plot(
            solution.saving_rates,
            n,
            color=cmap(i),
            label=r"$\langle s_i \rangle=$ "
            + str(
                np.around(
                    np.dot(solution.saving_rates, n) / solution.number_of_agents, 2
                )
            ),
        )

    axs.set_title("Distribution of Saving Rates")
    axs.set_xlabel(r"saving rates $s_l$")
    axs.set_ylabel(r"occupation numbers $n_l$")
    axs.set_yscale("log")
    axs.set_xscale("symlog")
    axs.legend()
    axs.grid()

    plt.show()


    # calculat the the production and coeffiecent of variation
    production = np.zeros(len(solutions))
    coef_var = np.zeros(len(solutions))
    mean_saving_rates = np.zeros(len(solutions))
    for i, solution in enumerate(solutions):
        # I'm done reversing##########################3
        moms = solution.exploration
        means = moms[0]
        n = solution.initial_occupation_numbers
        mean_saving_rates[i] = (
            np.dot(n, solution.saving_rates) / solution.number_of_agents
        )
        production[i] = psf.production(
            means, solution.elasticities, n, solution.labour, solution.number_of_agents
        )

        mean, var, _ = psf.moments_of_entire_population(
            moms, n, solution.number_of_agents
        )
        coef_var[i] = np.sqrt(var) / mean

    _, axs = plt.subplots(1, 2)
    axs[0].plot(mean_saving_rates, production, color="k", lw=3)
    axs[0].set_title("Production")
    axs[0].set_xlabel(r"Mean Saving Rate $\langle s_i \rangle$")
    axs[0].set_ylabel(r"Economic production $Y$")

    axs[0].grid()

    axs[1].set_title(
        "Effect of the Mean Saving Rate on Production and Wealth Inequality"
    )
    axs[1].plot(mean_saving_rates, coef_var, color="k", lw=3, label=f"Capital")
    axs[1].set_xlabel(r"Mean Saving Rate $\langle s_i \rangle$")
    axs[1].set_ylabel(r"CV")
    axs[1].grid()
    plt.show()


def plot_fluxes(fluxes):
    """
    Splits the positive and negative fluxes into and plots them into different layers of a pie chart.
    Since we seperate the incoming and outgoig fluxes, we need to add a filler element to compensate
    for the missing parts in each pie chart. For the Figure in the paper these were removed by hand afterwards.
    """
    in_fluxes = []
    out_fluxes = []
    for flux in fluxes:
        if flux > 0:
            in_fluxes.append(flux)
        else:
            out_fluxes.append(flux)
    norm = np.sum([np.array(in_fluxes)]) - np.sum(np.array(out_fluxes))
    in_fluxes.insert(0, norm - np.sum(np.array(in_fluxes)))
    out_fluxes.append(-norm - np.sum(np.array(out_fluxes)))

    _, ax = plt.subplots(figsize=(4.4, 4.4))

    size = 0.3

    labels_outer = [str(np.round(val,2)) for val in out_fluxes]
    labels_outer[-1] = "filler"
    labels_inner = [str(np.round(val,2)) for val in in_fluxes]
    labels_inner[0] = "filler"

    ax.pie(
        in_fluxes,
        radius=1,  # colors=outer_colors,
        wedgeprops=dict(width=size, edgecolor="k"),
        labels=labels_inner,
    )

    ax.pie(
        -np.array(out_fluxes),
        radius=1 - size,  # colors=inner_colors,
        wedgeprops=dict(width=size, edgecolor="k"),
        labels=labels_outer,
    )

    ax.set(aspect="equal")
    ax.legend()
    plt.show()
