import numpy as np
from sigmoidal_transition_prob import Economic_Process_sigmoidal
import plot_libarry as plt_lib


def fig8cd():
    tau = 300
    mean = 100
    var = 30
    sdev = np.sqrt(var)
    skewness = 0
    third_mom = sdev**3 * skewness + 3 * mean * sdev**2 + mean**3
    num_levels = 50
    num_of_agents = 500**2
    initial_moments = np.array([
        [mean for _ in range(num_levels)],
        [var for _ in range(num_levels)],
        [third_mom for _ in range(num_levels)],
    ])

    beta = 30
    # =====================================================================================================================
    processes = []
    epsilons = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    for exploration in epsilons:
        Process = Economic_Process_sigmoidal(
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
            exploration=exploration,
            gamma=1,
        )
        Process.solve_economics(4_000)
        processes.append(Process)
    plt_lib.plot_SM_fig8cd(processes)

def fig8ab():
    tau = 300
    mean = 100
    var = 30
    sdev = np.sqrt(var)
    skewness = 0
    third_mom = sdev**3 * skewness + 3 * mean * sdev**2 + mean**3
    num_levels = 50
    num_of_agents = 500**2
    initial_moments = np.array([
        [mean for _ in range(num_levels)],
        [var for _ in range(num_levels)],
        [third_mom for _ in range(num_levels)],
    ])

    exploration = 0.05
    # =====================================================================================================================
    processes = []
    betas = [2,5,10,20,30,50,60,70]
    for beta in betas:
        Process = Economic_Process_sigmoidal(
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
            exploration=exploration,
            gamma=1,
        )
        Process.solve_economics(4_000)
        processes.append(Process)
    plt_lib.plot_SM_fig8ab(processes)

def main():
    tau = 300
    mean = 100
    var = 30
    sdev = np.sqrt(var)
    skewness = 0
    third_mom = sdev**3 * skewness + 3 * mean * sdev**2 + mean**3
    num_levels = 30
    num_of_agents = 500**2
    initial_moments = np.array([
        [mean for _ in range(num_levels)],
        [var for _ in range(num_levels)],
        [third_mom for _ in range(num_levels)],
    ])

    beta = 30
    ex = 0.05
    # =====================================================================================================================
    processes_poverty = []
    # epsilons = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    betas = [2,5,10,20,30,50,60,70]
    for beta in betas:
        Process = Economic_Process_sigmoidal(
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
            exploration=ex,
            gamma=1,
        )
        Process.solve_economics(4_000)
        processes_poverty.append(Process)
    plt_lib.line_plot_saving_rate_distribution(processes_poverty)
    # Process.solve_economics(3_000)
    # Process.continue_solving()
    # =====================================================================================================================
    # high_saver_states = []
    # start_levels = 5
    # for beta in [30, 40, 50]:
    #     temp = Economic_Process_sigmoidal(
    #         number_of_agents=num_of_agents,
    #         number_of_levels=start_levels,
    #         update_time=tau,
    #         temperature=1 / beta,
    #         initial_occupation_numbers=np.full(
    #             start_levels, num_of_agents / start_levels
    #         ),
    #         initial_capital_moments=[
    #             [mean for _ in range(start_levels)],
    #             [var for _ in range(start_levels)],
    #             [third_mom for _ in range(start_levels)],
    #         ],
    #         labour=num_of_agents,
    #         elasticities=[0.5, 0.5],
    #         production_constant=1,
    #         depreciation=0.05,
    #         exploration=0.05,
    #         gamma=1,
    #     )
    #     temp.solve_SDE_economics(30_000, 1)
    #     plt_lib.plot_simple_time_series(temp)
    #     time = int(input("Enter the time"))
    #     new_n = temp.unpacking_occupation_numbers()[time]
    #     means = temp.capital_moments[time][0]
    #     vars = temp.capital_moments[time][1]
    #     third_moms = temp.capital_moments[time][2]
    #     levels = 6
    #     old_saving_rate = temp.saving_rates
    #     processes_temp = []
    #     while levels < num_levels:
    #         s_interp = np.linspace(0.05, 0.95, levels)
    #         new_n = (
    #             500
    #             * np.interp(s_interp, old_saving_rate, new_n)
    #             / np.sum(np.interp(s_interp, old_saving_rate, new_n))
    #         )
    #         print(levels)
    #
    #         temp = Economic_Process_sigmoidal(
    #             number_of_agents=num_of_agents,
    #             number_of_levels=levels,
    #             update_time=tau,
    #             temperature=1 / beta,
    #             initial_occupation_numbers=new_n,
    #             initial_capital_moments=np.array([
    #                 np.sum(means)
    #                 * np.interp(s_interp, old_saving_rate, means)
    #                 / np.sum(np.interp(s_interp, old_saving_rate, means)),
    #                 np.interp(s_interp, old_saving_rate, vars),
    #                 np.interp(s_interp, old_saving_rate, third_moms),
    #             ]),
    #             labour=num_of_agents,
    #             elasticities=[0.5, 0.5],
    #             production_constant=1,
    #             depreciation=0.05,
    #             exploration=0.05,
    #             gamma=1,
    #         )
    #         temp.solve_economics(1000)
    #         old_saving_rate = temp.saving_rates
    #         new_n = temp.unpacking_occupation_numbers()[-1]
    #         means = temp.capital_moments[-1][0]
    #         vars = temp.capital_moments[-1][1]
    #         third_moms = temp.capital_moments[-1][2]
    #         levels += 1
    #         processes_temp.append(temp)
    #     plt_lib.line_plot_saving_rate_distribution(processes_temp)
    #     high_saver_states.append(temp)
    #
    # plt_lib.line_plot_saving_rate_distribution(high_saver_states)


if __name__ == "__main__":
    fig8ab()
    fig8cd()
