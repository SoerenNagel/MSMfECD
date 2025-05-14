import matplotlib.pyplot as plt
import numpy as np
from sigmoidal_transition_prob import Economic_Process_sigmoidal
import plot_libarry as plt_lib
import phase_space_functions as psf
import pickle
import tqdm


def analyse_fixed_points(process, betas, states=None):
    """
    This function takes a process, plots the saving rate distribution.
    Then asks you for the relevant time_steps.
    The states found at the relevant times will be checkd with
    an adiabatic variation of beta. This will trace the stable state
    and show us any transitions.
    Since the SDE seitches between the fixed points of the ODE, this gives us intial conditions, 
    for which the ODE.
    """
    if states is None:
        plt_lib.plot_saving_rate_color_plot(process)
        states = []
        a = int(input("Number of obseved states:"))
        for i in range(a):
            states.append(int(input(f"Time {i+1}:")))
    fixed_points_n = np.zeros((len(states), len(betas), process.number_of_levels))
    fixed_points_moments = np.zeros(
        (len(states), len(betas), 3, process.number_of_levels)
    )

    # for all the meta stable states, we now check the stability in of the ODE
    # for the given range of betas by solving the ODE until we reach a fixed point
    # and then take that new quilibrium as initial data for the next integration,
    # with shifted beta
    for i, time in enumerate(states):
        #get the initial_data from the SDE solution
        new_initial_n = process.unpacking_occupation_numbers()[time]
        new_initial_moments = process.capital_moments[time]
        #set up a new process
        diff_eq = Economic_Process_sigmoidal(
            number_of_agents=2300,
            number_of_levels=process.number_of_levels,
            update_time=process.update_time,
            temperature=process.temperature,
            initial_occupation_numbers=new_initial_n,
            initial_capital_moments=new_initial_moments,
            labour=2300,
            elasticities=process.elasticities,
            production_constant=process.production_constant,
            depreciation=process.depreciation,
            exploration=process.exploration,
            gamma=1,
        )
        # solve the ODE the first time for a very large T to allow for equilibriation
        diff_eq.solve_economics(100_000)
        # set the end of the simulation as new intial data
        new_initial_n = diff_eq.unpacking_occupation_numbers()[-1]
        new_initial_moments = diff_eq.capital_moments[-1]
        #################################################################################################################

        for j, beta in tqdm.tqdm(enumerate(betas)):
            fixed_points_n[i,j,:] = new_initial_n
            fixed_points_moments[i,j,:,:] = new_initial_moments
            for level in range(process.number_of_levels):
                fixed_points_n[i][j][level] = new_initial_n[level]
                for mom in range(3):
                    fixed_points_moments[i][j][mom][level] = new_initial_moments[mom][level]
            differential_eq = Economic_Process_sigmoidal(
                number_of_agents=2300,
                number_of_levels=process.number_of_levels,
                update_time=process.update_time,
                temperature=1 / beta,
                initial_occupation_numbers=new_initial_n,
                initial_capital_moments=new_initial_moments,
                labour=2300,
                elasticities=process.elasticities,
                production_constant=process.production_constant,
                depreciation=process.depreciation,
                exploration=process.exploration,
                gamma=1,
            )
            differential_eq.solve_economics(20_000)
            new_initial_n = differential_eq.unpacking_occupation_numbers()[-1]
            new_initial_moments = differential_eq.capital_moments[-1]


    return [states, fixed_points_n, fixed_points_moments]


def generate_figures():
    """
    This function reads the saved data for the bifurcation diagram and genrates Fig1b from the paper,
    aswell as Fig4 and Fig7 from the Supplementary Material.
    """
    file = "data/beta-bifurcation_30_Levels"
    with open(file, "rb") as f:
        bif_data = pickle.load(f)
    bif, betas, process, bt_bif = bif_data

    # Plot the actual bifurcation diagram
    plt_lib.plot_bifurcation_diagramm(bif, betas, process, back_tracking=bt_bif)
    _, fixed_n, fixed_m = bif
    _, bt_fixed_n, bt_fixed_m, _ = bt_bif

    # For each state, we take the first element from the beta axis,
    # and since the "bt_" were gathered, with reversed beta we take the last set of occupation numbers.
    ns = np.concatenate([fixed_n[:,0,:], bt_fixed_n[:, -1, :]])
    moments = np.concatenate([fixed_m[:,0, :, :], bt_fixed_m[:, -1, :, :]])

    processes_for_plot = []
    for n,m in zip(ns, moments):
        processes_for_plot.append(
            Economic_Process_sigmoidal(
                2300,
                process.number_of_levels,
                process.update_time,
                1 / 50,
                n,
                m,
                process.labour,
                process.elasticities,
                process.production_constant,
                process.depreciation,
                process.exploration,
                gamma=1,
            )
        )
    plt_lib.plot_analayse_states_from_bifurcation_diagram(processes_for_plot)


######################################################################################################################
def new_main():
    # this makes use of the fact, that the ode is invariant under changes in N, as long as N/L= constant.
    # so we can take initialdata from any metastable state of the ode and will likely be close to a staedy state.
    # we will integrate the ODE later on, to ensure we are only recording steady states.
    betas = np.linspace(50, 0.6, 800)
    file = f"data/tau=300, Agents 2300, Temp = 0.02Levels=30 2nd_run"
    with open(file, "rb") as f:
        process = pickle.load(f)
    bifurcation_data = analyse_fixed_points(process, betas, states=[1000, 10000, 25000, 40000, 60000])
    file2 = f"data/tau=300, Agents 2300, Temp = 0.02Levels=30"
    with open(file2, "rb") as f:
        process = pickle.load(f)
    bifurcation_data2 = analyse_fixed_points(process, betas, states=[3000,6000, 17000, 30000, 10000])
    file2 = f"data/tau=300, Agents 2200, Temp = 0.02Levels=30"
    with open(file2, "rb") as f:
        process = pickle.load(f)
    bifurcation_data3 = analyse_fixed_points(process, betas, states=[1000, 8000])

    states = np.concatenate(
        (bifurcation_data[0], bifurcation_data2[0], bifurcation_data3[0])
    )
    fixed_n = np.concatenate(
        (bifurcation_data[1], bifurcation_data2[1], bifurcation_data3[1])
    )
    fixed_m = np.concatenate(
        (bifurcation_data[2], bifurcation_data2[2], bifurcation_data3[2])
    )

    #######################################################################################################################

    _, axs = plt.subplots()
    for state_num in range(len(states)):
        mean_saving_rate = np.zeros(len(betas))
        prod = np.zeros(len(betas))
        for j, _ in enumerate(betas):
            mean_saving_rate[j] = (
                np.dot(process.saving_rates, fixed_n[state_num][j])
                / process.number_of_agents
            )
            prod[j] = psf.production(
                fixed_m[state_num][j][0],
                process.production_constant,
                fixed_n[state_num][j],
                process.elasticities,
                process.labour,
            )
        axs.plot(
            range(len(betas)),
            mean_saving_rate,
            lw=2,
            label=f"state_num={states[state_num]}",
        )
    axs.set_ylabel(r"Mean Saving Rate $\langle s_i \rangle$")
    axs.set_xlabel(r"$\beta$")
    axs.grid(True)
    axs.legend()
    plt.show()
    #######################################################################################################################
    size = int(input("Size of array:"))
    back_tracking_in = np.zeros((size, 2))
    for i in range(size):
        back_tracking_in[i][0] = float(input(f"State {i+1}:"))
        back_tracking_in[i][1] = float(input(f"Beta_Index {i+1}:"))
    back_tracking_result = []
    bt_betas = []
    for state, beta_index in back_tracking_in:
        beta_index = int(beta_index)
        state_index = int(np.where(states == state)[0][0])
        new_betas = np.linspace(betas[beta_index], 50, 500)
        bt_betas.append(new_betas)
        initial_occ = fixed_n[state_index][beta_index]
        initial_mom = fixed_m[state_index][beta_index]
        back_tack = Economic_Process_sigmoidal(
            number_of_agents=2300,
            number_of_levels=process.number_of_levels,
            update_time=process.update_time,
            temperature=process.temperature,
            initial_occupation_numbers=initial_occ,
            initial_capital_moments=initial_mom,
            labour=process.labour,
            elasticities=process.elasticities,
            production_constant=process.production_constant,
            depreciation=process.depreciation,
            exploration=process.exploration,
            gamma=1,
        )
        back_tack.solve_economics(len(back_tracking_in) + 100)
        back_tracking_result.append(analyse_fixed_points(back_tack, new_betas))

    bt_states = np.concatenate(
        (np.array([back_tracking_result[k][0] for k in range(size)]))
    )
    bt_fixed_n = np.concatenate(
        (np.array([back_tracking_result[k][1] for k in range(size)]))
    )
    bt_fixed_m = np.concatenate(
        (np.array([back_tracking_result[k][2] for k in range(size)]))
    )
    bt_bif = [bt_states, bt_fixed_n, bt_fixed_m, bt_betas]

    file = "beta-bifurcation_30_Levels"
    with open(file, "wb") as f:
        pickle.dump([[states, fixed_n, fixed_m], betas, process, bt_bif], f)

    #######################################################################################################################
    plt_lib.plot_bifurcation_diagramm(
        [states, fixed_n, fixed_m], betas, process, back_tracking=bt_bif
    )
    processes_for_plot = []
    for i, _ in enumerate(states):
        processes_for_plot.append(
            Economic_Process_sigmoidal(
                2300,
                process.number_of_levels,
                process.update_time,
                1 / 50,
                fixed_n[i][0],
                fixed_m[i][0],
                process.labour,
                process.elasticities,
                process.production_constant,
                process.depreciation,
                process.exploration,
                gamma=1,
            )
        )
    for i, _ in enumerate(bt_states):
        processes_for_plot.append(
            Economic_Process_sigmoidal(
                2300,
                process.number_of_levels,
                process.update_time,
                1 / 50,
                fixed_n[i][-1],
                fixed_m[i][-1],
                process.labour,
                process.elasticities,
                process.production_constant,
                process.depreciation,
                process.exploration,
                gamma=1,
            )
        )
    plt_lib.plot_analayse_states_from_bifurcation_diagram(processes_for_plot)



    file2 = f"data/tau=300, Agents 2200, Temp = 0.02Levels=30"
    with open(file2, "rb") as f:
        process = pickle.load(f)
    bifurcation_data3 = analyse_fixed_points(process, betas)

    states = np.concatenate(
        (bifurcation_data[0], bifurcation_data2[0], bifurcation_data3[0])
    )
    fixed_n = np.concatenate(
        (bifurcation_data[1], bifurcation_data2[1], bifurcation_data3[1])
    )
    fixed_m = np.concatenate(
        (bifurcation_data[2], bifurcation_data2[2], bifurcation_data3[2])
    )
    #######################################################################################################################

    _, axs = plt.subplots()
    for state_num in range(len(states)):
        mean_saving_rate = np.zeros(len(betas))
        for j, _ in enumerate(betas):
            mean_saving_rate[j] = (
                np.dot(process.saving_rates, fixed_n[state_num][j])
                / process.number_of_agents
            )
        axs.plot(
            range(len(betas)),
            mean_saving_rate,
            lw=2,
            label=f"state_num={states[state_num]}",
        )
    axs.set_ylabel(r"Mean Saving Rate $\langle s_i \rangle$")
    axs.set_xlabel(r"$\beta$")
    axs.grid(True)
    axs.legend()
    plt.show()
    #######################################################################################################################
    size = int(input("Number of new states found, that could have a stable branch:"))
    back_tracking_in = np.zeros((size, 2))
    for i in range(size):
        back_tracking_in[i][0] = float(input(f"State {i+1}:"))
        back_tracking_in[i][1] = float(input(f"Beta_Index {i+1}:"))
    back_tracking_result = []
    bt_betas = []
    for state, beta_index in back_tracking_in:
        beta_index = int(beta_index)
        state_index = int(np.where(states == state)[0][0])
        new_betas = np.linspace(betas[beta_index], 50, 500)
        bt_betas.append(new_betas)
        initial_occ = fixed_n[state_index][beta_index]
        initial_mom = fixed_m[state_index][beta_index]
        back_tack = Economic_Process_sigmoidal(
            number_of_agents=2300,
            number_of_levels=process.number_of_levels,
            update_time=process.update_time,
            temperature=process.temperature,
            initial_occupation_numbers=initial_occ,
            initial_capital_moments=initial_mom,
            labour=process.labour,
            elasticities=process.elasticities,
            production_constant=process.production_constant,
            depreciation=process.depreciation,
            exploration=process.exploration,
            gamma=1,
        )
        back_tack.solve_economics(len(back_tracking_in) + 100)
        back_tracking_result.append(analyse_fixed_points(back_tack, new_betas))

    bt_states = np.concatenate(
        (np.array([back_tracking_result[k][0] for k in range(size)]))
    )
    bt_fixed_n = np.concatenate(
        (np.array([back_tracking_result[k][1] for k in range(size)]))
    )
    bt_fixed_m = np.concatenate(
        (np.array([back_tracking_result[k][2] for k in range(size)]))
    )
    bt_bif = [bt_states, bt_fixed_n, bt_fixed_m, bt_betas]
    
    file = "beta-bifurcation_30_Levels"
    with open(file, "wb") as f:
        pickle.dump([[states, fixed_n, fixed_m], betas, process, bt_bif], f)

    #######################################################################################################################
    plt_lib.plot_bifurcation_diagramm(
        [states, fixed_n, fixed_m], betas, process, back_tracking=bt_bif
    )
    processes_for_plot = []
    for i, _ in enumerate(states):
        processes_for_plot.append(
            Economic_Process_sigmoidal(
                2300,
                process.number_of_levels,
                process.update_time,
                1 / 50,
                fixed_n[i][0],
                fixed_m[i][0],
                process.labour,
                process.elasticities,
                process.production_constant,
                process.depreciation,
                process.exploration,
                gamma=1,
            )
        )
    for i, _ in enumerate(bt_states):
        processes_for_plot.append(
            Economic_Process_sigmoidal(
                2300,
                process.number_of_levels,
                process.update_time,
                1 / 50,
                fixed_n[i][-1],
                fixed_m[i][-1],
                process.labour,
                process.elasticities,
                process.production_constant,
                process.depreciation,
                process.exploration,
                gamma=1,
            )
        )
    plt_lib.plot_analayse_states_from_bifurcation_diagram(processes_for_plot)


if __name__ == "__main__":
    # new_main()
    generate_figures()
