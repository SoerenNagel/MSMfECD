import numba
import math
import numpy as np


@numba.njit(cache=True)
def aggregate_capital(occupation_numbers, capital_means):
    """
    Compute the total aggregated capital as the dot product of occupation numbers and capital means.

    Parameters
    ----------
    occupation_numbers : ndarray
        A 1d array representing the number of agents in each saving rate level.
    capital_means : ndarray
        A 1d array representing the mean capital of agents in each aving rate level.

    Returns
    -------
    float
        The total aggregated capital.
    """
    return np.dot(occupation_numbers, capital_means)


@numba.njit(cache=True)
def return_on_investment(
    capital_means, production_constant, elasticities, labour, occupation_numbers
):
    """
    Compute the return on investment (ROI) based on a Cobb-Douglas production function.

    Parameters
    ----------
    capital_means : ndarray
        An array representing the mean capital of agents in each saving rate level.
    production_constant : float
        A constant scaling factor in the production function.
    elasticities : ndarray
        A two-element array containing the capital and labor elasticities.
    labour : float
        The total labor input.
    occupation_numbers : ndarray
        An array representing the number of agents in each saving rate level.

    Returns
    -------
    float
        The computed return on investment.
    """
    roi = (
        production_constant
        * elasticities[0]
        * np.dot(occupation_numbers, capital_means) ** (elasticities[0] - 1)
        * labour ** elasticities[1]
    )
    return roi


@numba.njit(cache=True)
def production(
    capital_means, production_constant, occupation_numbers, elasticities, labour
):
    """
    Compute the total production based on a Cobb-Douglas production function.

    Parameters
    ----------
    capital_means : ndarray
        An array representing the mean capital of agents at different saving rate levels.
    production_constant : float
        A constant scaling factor in the production function.
    occupation_numbers : ndarray
        An array representing the number of agents at different saving rate levels.
    elasticities : tuple or ndarray
        A two-element array or tuple containing the capital and labor elasticities.
    labour : float
        The total labor input.

    Returns
    -------
    float
        The total production output.
    """
    return (
        production_constant
        * np.dot(occupation_numbers, capital_means) ** elasticities[0]
        * labour ** elasticities[1]
    )


@numba.njit(cache=True)
def wages(occupation_numbers, capital_means, production_constant, elasticities, labour):
    """
    Compute wages based on a Cobb-Douglas production function.

    Parameters
    ----------
    occupation_numbers : ndarray
        An array representing the number of agents at different saving rate levels.
    capital_means : ndarray
        An array representing the mean capital of agents at different saving rate levels.
    production_constant : float
        A constant scaling factor in the production function.
    elasticities : tuple or ndarray
        A two-element array or tuple containing the capital and labor elasticities.
    labour : float
        The total labor input.

    Returns
    -------
    float
        The computed wage level.
    """
    wages = (
        production_constant
        * elasticities[1]
        * labour ** (elasticities[1] - 1)
        * np.dot(occupation_numbers, capital_means) ** elasticities[0]
    )
    return wages


def income_of_level(
    occupation_numbers, capital_means, production_constant, elasticities, labour, level
):
    """
    Compute the income of agents at a given saving rate level.

    The income consists of capital returns and wage earnings.

    Parameters
    ----------
    occupation_numbers : ndarray
        An array representing the number of agents at different saving rate levels.
    capital_means : ndarray
        An array representing the mean capital of agents at different saving rate levels.
    production_constant : float
        A constant scaling factor in the production function.
    elasticities : tuple or ndarray
        A two-element array or tuple containing the capital and labor elasticities.
    labour : float
        The total labor input.
    level : int
        The index of the saving rate level for which income is computed.

    Returns
    -------
    float
        The computed income of agents at the given saving rate level.
    """
    r = return_on_investment(
        capital_means, production_constant, elasticities, labour, occupation_numbers
    )
    w = wages(
        occupation_numbers, capital_means, production_constant, elasticities, labour
    )
    N = np.sum(occupation_numbers)
    return r * capital_means[level] + w * labour / N


@numba.njit(cache=True)
def moments_of_consumption_distribution(
    capital_means,
    capital_variances,
    capital_3rd_mom,
    occupation_numbers,
    production_constant,
    elasticities,
    labour,
    number_of_levels,
    saving_rates,
    number_of_agents,
):
    """
    Compute the first three moments (mean, variance, third moment) of the consumption distribution.

    Parameters
    ----------
    capital_means : ndarray
        Mean capital of agents at each saving rate level.
    capital_variances : ndarray
        Variance of capital at each saving rate level.
    capital_3rd_mom : ndarray
        Third (non-centralized) moment of capital at each saving rate level.
    occupation_numbers : ndarray
        Number of agents at different saving rate levels.
    production_constant : float
        A constant scaling factor in the production function.
    elasticities : tuple or ndarray
        A two-element array or tuple containing the capital and labor elasticities.
    labour : float
        The total labor input.
    number_of_levels : int
        The number of saving rate levels.
    saving_rates : ndarray
        An array of saving rates corresponding to different levels.
    number_of_agents : int
        The total number of agents.

    Returns
    -------
    ndarray
        A 2D array of shape (3, number_of_levels), where:
        - Row 0 contains the mean consumption at each saving rate level.
        - Row 1 contains the variance of consumption at each saving rate level.
        - Row 2 contains the third moment (not centralized) of consumption at each saving rate level.
    """
    w = (
        production_constant
        * elasticities[1]
        * labour ** (elasticities[1] - 1)
        * np.dot(occupation_numbers, capital_means) ** elasticities[0]
    )
    roi = (
        production_constant
        * elasticities[0]
        * np.dot(occupation_numbers, capital_means) ** (elasticities[0] - 1)
        * labour ** elasticities[1]
    )

    consumption_moments = np.zeros((3, number_of_levels))

    for l in range(number_of_levels):
        consumption_moments[0][l] = (1 - saving_rates[l]) * (
            roi * capital_means[l] + w * labour / number_of_agents
        )
        consumption_moments[1][l] = (
            (1 - saving_rates[l]) ** 2 * roi**2 * capital_variances[l]
        )
        # here we have to refactor again from the variance to the second moment mu2
        mu2 = capital_variances[l] + capital_means[l] ** 2
        consumption_moments[2][l] = (1 - saving_rates[l]) ** 3 * (
            roi**3 * capital_3rd_mom[l]
            + 3 * roi * capital_means[l] * w**2 * labour**2 / number_of_agents**2
            + 3 * w * (labour / number_of_agents) * roi**2 * mu2
            + (w * labour / number_of_agents) ** 3
        )
    return consumption_moments


def moments_of_entire_population(moments, occupaion_numbers, num_agents):
    """
    Compute the first three moments (mean, variance, third moment) of the entire population.

    Parameters
    ----------
    moments : array of ndarrays
        A array containing three arrays:
        - means: The mean values at different saving rate levels.
        - vars: The variances at different saving rate levels.
        - third_moms: The third moments (not centralized) at different saving rate levels.
    occupation_numbers : ndarray
        An array representing the number of agents at different saving rate levels.
    num_agents : int
        The total number of agents.

    Returns
    -------
    ndarray
        A 1D array containing:
        - The mean of the entire population.
        - The variance of the entire population.
        - The third moment (not centralized) of the entire population.
    """
    means, vars, third_moms = moments
    total_mean = np.dot(occupaion_numbers, means) / num_agents
    total_var = np.dot(occupaion_numbers, vars + means * means) - total_mean
    total_third_mom = np.dot(occupaion_numbers, third_moms) / num_agents
    return np.array([total_mean, total_var, total_third_mom])


def moments_of_entire_population_income(
    moments,
    occupation_numbers,
    num_agents,
    saving_rates,
    production_constant,
    elasticities,
    labour,
    number_of_levels,
):
    """
    Compute the first three moments (mean, variance, third moment) of the entire population's income distribution.

    This function first derives the income distribution moments from the consumption distribution moments,
    then aggregates them over the entire population.

    Parameters
    ----------
    moments : array of ndarrays
        A tuple containing three arrays:
        - means: The mean consumption at different saving rate levels.
        - vars: The variance of consumption at different saving rate levels.
        - third_moms: The third moment (not centralized) of consumption at different saving rate levels.
    occupation_numbers : ndarray
        An array representing the number of agents at different saving rate levels.
    num_agents : int
        The total number of agents.
    saving_rates : ndarray
        An array of saving rates corresponding to different levels.
    production_constant : float
        A constant scaling factor in the production function.
    elasticities : tuple or ndarray
        A two-element array or tuple containing the capital and labor elasticities.
    labour : float
        The total labor input.
    number_of_levels : int
        The number of saving rate levels.

    Returns
    -------
    ndarray
        A 1D array containing:
        - The mean income of the entire population.
        - The variance of income of the entire population.
        - The third moment (not centralized) of income of the entire population.
    """
    means, vars, third_moms = moments_of_consumption_distribution(
        moments[0],
        moments[1],
        moments[2],
        occupation_numbers,
        production_constant,
        elasticities,
        labour,
        number_of_levels,
        saving_rates,
        num_agents,
    )
    income_means = means / (1 - saving_rates)
    income_vars = vars / ((1 - saving_rates) ** 2)
    income_third_moms = third_moms / (1 - saving_rates**3)
    return moments_of_entire_population(
        [income_means, income_vars, third_moms], occupation_numbers, num_agents
    )


########################################################################################################################
# dynamical equations


@numba.njit
def expectations_temp_exp(con_moments, temperature, number_of_levels):
    """
    Compute the expectation of the exponential of consumption moments.

    This function calculates E[exp(consumption/temperature)] using the first two moments
    of the consumption distribution, assuming a Gaussian approximation.

    Parameters
    ----------
    con_moments : ndarray
        A 2D array where:
        - Row 0 contains the mean consumption at each saving rate level.
        - Row 1 contains the variance of consumption at each saving rate level.
    temperature : float
        A scaling factor in the exponential function.
    number_of_levels : int
        The number of saving rate levels.

    Returns
    -------
    ndarray
        An array of shape (number_of_levels,) containing the computed expectations.
    """
    result = np.zeros(number_of_levels)
    for level in range(number_of_levels):
        result[level] = np.exp(
            (
                temperature ** (-2) * con_moments[1][level]
                + 2 * con_moments[0][level] * temperature ** (-1)
            )
            / 2
        )
    return result


@numba.njit(cache=True)
def expectations_temp_exp_general(con_moments, temperature, number_of_levels):
    """
    Compute the expectation of the exponential of consumption moments using a third-order expansion.

    This function calculates an approximation of E[exp(consumption/temperature)] by including 
    the mean, variance, and third central moment of the consumption distribution.

    Parameters
    ----------
    con_moments : ndarray
        A 2D array where:
        - Row 0 contains the mean consumption at each saving rate level.
        - Row 1 contains the variance of consumption at each saving rate level.
        - Row 2 contains the third (non-centralized) moment of consumption at each saving rate level.
    temperature : float
        A scaling factor in the exponential function.
    number_of_levels : int
        The number of saving rate levels.

    Returns
    -------
    ndarray
        An array of shape (number_of_levels,) containing the computed expectations.
    """
    result = np.zeros(number_of_levels)
    for level in range(number_of_levels):
        # result[level] = 1 + con_moments[0][level]/temperature + (con_moments[1][level] + con_moments[0][level] ** 2) / (2 * temperature ** 2) + con_moments[2][level]/(6 * temperature ** 3)
        third_central_mom = (
            con_moments[2][level]
            - 3 * con_moments[0][level] * con_moments[1][level]
            - con_moments[0][level]
        )
        result[level] = np.exp(con_moments[0][level] / temperature) * (
            1
            + con_moments[1][level] / (2 * temperature**2)
            + third_central_mom / (6 * temperature**3)
        )

    return result


@numba.njit(cache=True)
def sigmoidal_dynamical_equations(
    _,
    y,
    number_of_levels,
    number_of_agents,
    production_constant,
    elasticities,
    labour,
    saving_rates,
    update_time,
    depreciation,
    transition_vectors,
    temperature,
    exploration,
):

    # Splitting the moments and occupation numbers and applying the exponential, because we are integrating the log
    # in order to avoid negative values. We do the same for the variance of the capital stocks.
    # From this we pick up additional terms, from the appropriate transformations.
    log_n = y[-number_of_levels:]
    n = np.exp(log_n)

    means = y[:number_of_levels]
    variances = np.exp(y[number_of_levels : 2 * number_of_levels])
    second_moments = variances + means * means
    capital_3rd_moments = y[2 * number_of_levels : 3 * number_of_levels]
    ###############################################
    # catching infinities
    lower_bound = 10**-20
    for i in range(number_of_levels):
        if n[i] < lower_bound:
            n[i] = lower_bound
        if variances[i] < lower_bound:
            variances[i] = lower_bound

    ################################################

    con_mom = moments_of_consumption_distribution(
        means,
        variances,
        capital_3rd_moments,
        n,
        production_constant,
        elasticities,
        labour,
        number_of_levels,
        saving_rates,
        number_of_agents,
    )  # uses list[nth-moment][level]

    w = wages(n, means, production_constant, elasticities, labour)
    roi = return_on_investment(means, production_constant, elasticities, labour, n)
    s = saving_rates
    L = labour
    N = number_of_agents
    tau = update_time
    delta = depreciation
    ################################################################################################################
    # dynamical equations for the occupation numbers
    expects = expectations_temp_exp_general(con_mom, temperature, number_of_levels)
    partition_func = np.sum(n * expects)
    dn = np.zeros(number_of_levels)
    dn_explor = np.zeros(number_of_levels)
    for i in range(number_of_levels):
        for j in range(number_of_levels):
            dn += n[i] * n[j] * transition_vectors[i][j] * expects[j]
            dn_explor += n[i] * transition_vectors[i][j]
    dn = 1 / (tau * partition_func) * dn
    dn_explor = 1 / (tau * number_of_levels) * dn_explor
    # Here we combine the exploration and immitating behaviour and divide by n for the change of log(n)
    dn_total = (1 - exploration) * dn + exploration * dn_explor

    d_log_n = dn_total / n
    # This are the dynamical equations for the moments!

    delta_means = np.zeros(number_of_levels)
    delta_vars = np.zeros(number_of_levels)
    delta_3rd_moments = np.zeros(number_of_levels)

    for l in range(number_of_levels):
        delta_means[l] = (roi * s[l] - delta) * means[l] + w * s[l] * L / N
        delta_vars[l] = 2 * (roi * s[l] - delta) * variances[l]
        delta_3rd_moments[l] = (
            3 * (roi * s[l] - delta) * capital_3rd_moments[l]
            + 3 * w * s[l] * L * (variances[l] + means[l] ** 2) / N
        )

    ################################################################################################################
    # adding the jump corrections for the first moment
    individual_mean_jump_corrections = np.zeros((number_of_levels, number_of_levels))
    ind_mean_j_corr_explore = np.zeros((number_of_levels, number_of_levels))
    individual_var_jump_corrections = np.zeros((number_of_levels, number_of_levels))
    ind_var_j_corr_explore = np.zeros((number_of_levels, number_of_levels))
    individual_3rd_mom_jump_corrections = np.zeros((number_of_levels, number_of_levels))
    ind_3rd_mom_j_corr_explore = np.zeros((number_of_levels, number_of_levels))
    # we have to do this three times, due to numba:
    for l in range(number_of_levels):
        for k in range(number_of_levels):
            individual_mean_jump_corrections[l][k] = (
                (means[k] - means[l]) * n[k] * expects[l]
            )
            ind_mean_j_corr_explore[l][k] = (means[k] - means[l]) * n[k]
    # here we technically only add the jump correction for the second non-central moments,
    # going to the variance will only yield an extra term, that will be added later
    for l in range(number_of_levels):
        for k in range(number_of_levels):
            individual_var_jump_corrections[l][k] = (
                (second_moments[k] - second_moments[l]) * n[k] * expects[l]
            )
            ind_var_j_corr_explore[l][k] = (second_moments[k] - second_moments[l]) * n[
                k
            ]
    for l in range(number_of_levels):
        for k in range(number_of_levels):
            individual_3rd_mom_jump_corrections[l][k] = (
                (capital_3rd_moments[k] - capital_3rd_moments[l]) * n[k] * expects[l]
            )
            ind_3rd_mom_j_corr_explore[l][k] = (
                capital_3rd_moments[k] - capital_3rd_moments[l]
            ) * n[k]

    individual_mean_jump_corrections = individual_mean_jump_corrections / (
        tau * partition_func
    )
    individual_var_jump_corrections = individual_var_jump_corrections / (
        tau * partition_func
    )
    individual_3rd_mom_jump_corrections = individual_3rd_mom_jump_corrections / (
        tau * partition_func
    )

    ind_mean_j_corr_explore = ind_mean_j_corr_explore / (tau * number_of_levels * n)
    ind_var_j_corr_explore = ind_var_j_corr_explore / (tau * number_of_levels * n)
    ind_3rd_mom_j_corr_explore = ind_3rd_mom_j_corr_explore / (
        tau * number_of_levels * n
    )

    mean_jump_corrections = (1 - exploration) * np.sum(
        individual_mean_jump_corrections, axis=1
    ) + exploration * np.sum(ind_mean_j_corr_explore, axis=1)
    var_jump_corrections = (1 - exploration) * np.sum(
        individual_var_jump_corrections, axis=1
    ) + exploration * np.sum(ind_var_j_corr_explore, axis=1)
    capital_3rd_moment_jump_corrections = (1 - exploration) * np.sum(
        individual_3rd_mom_jump_corrections, axis=1
    ) + exploration * np.sum(ind_3rd_mom_j_corr_explore, axis=1)

    ##############################################################################################################
    rates = transition_rates(
        means,
        variances,
        capital_3rd_moments,
        n,
        update_time,
        exploration,
        number_of_levels,
        production_constant,
        elasticities,
        labour,
        saving_rates,
        N,
        temperature,
    )
    d_means_test = np.zeros(number_of_levels)
    d_vars_test = np.zeros(number_of_levels)
    d_third_moms_test = np.zeros(number_of_levels)
    for l in range(number_of_levels):
        d_means_test[l] = (roi * s[l] - delta) * means[l] + w * s[l] * L / N
        d_vars_test[l] = 2 * (roi * s[l] - delta)
        d_third_moms_test[l] = (
            3 * (roi * s[l] - delta) * capital_3rd_moments[l]
            + 3 * w * s[l] * L * (variances[l] + means[l] ** 2) / N
        )
    for l in range(number_of_levels):
        for i in range(number_of_levels):
            d_means_test[l] += (means[i] - means[l]) / n[l] * rates[i][l]
            d_third_moms_test[l] += (
                (capital_3rd_moments[i] - capital_3rd_moments[l]) / n[l] * rates[i][l]
            )
            d_vars_test[l] += (
                (second_moments[i] - second_moments[l])
                / n[l]
                * rates[i][l]
                / variances[l]
            )
            d_vars_test[l] += (
                -2
                * means[l]
                * (means[i] - means[l])
                / n[l]
                * rates[i][l]
                / variances[l]
            )
    ##############################################################################################################
    # her we add the additional term for the variance
    var_jump_corrections += -2 * means * mean_jump_corrections
    delta_means += mean_jump_corrections
    delta_vars += var_jump_corrections
    delta_3rd_moments += capital_3rd_moment_jump_corrections
    ################################################################################################################
    # putting moments and occupation numbers together
    delta_log_vars = np.zeros(number_of_levels)
    for level in range(number_of_levels):
        if variances[level] > 10**-10:
            delta_log_vars[level] = delta_vars[level] / variances[level]
        else:
            delta_log_vars[level] = 0
    delta_y = np.concatenate((delta_means, delta_log_vars, delta_3rd_moments, d_log_n))

    return delta_y


@numba.njit(cache=True)
def transition_rates(
    means,
    vars,
    third_moms,
    n,
    tau,
    exploration,
    number_of_levels,
    production_constant,
    elasticities,
    labour,
    saving_rates,
    number_of_agents,
    temperature,
):
    """
    Compute the transition rates between different saving rate levels.

    This function calculates the transition rates of agents moving between different 
    saving rate levels based on the moments of the consumption distribution and 
    a temperature-based exploration factor.

    Parameters
    ----------
    means : ndarray
        The mean capital levels at different saving rate levels.
    vars : ndarray
        The variance of capital at different saving rate levels.
    third_moms : ndarray
        The third moment (not centralized) of capital at different saving rate levels.
    n : ndarray
        The number of agents at each saving rate level.
    tau : float
        A characteristic timescale for transitions.
    exploration : float
        The probability of random exploration (0 ≤ exploration ≤ 1).
    number_of_levels : int
        The number of saving rate levels.
    production_constant : float
        A scaling constant in the production function.
    elasticities : tuple or ndarray
        A two-element array or tuple containing the capital and labor elasticities.
    labour : float
        The total labor input.
    saving_rates : ndarray
        An array of saving rates corresponding to different levels.
    number_of_agents : int
        The total number of agents.
    temperature : float
        A scaling factor affecting the transition probabilities.

    Returns
    -------
    ndarray
        A 2D array (number_of_levels × number_of_levels) containing the transition rates
        between saving rate levels.

    """
    c_moms = moments_of_consumption_distribution(
        means,
        vars,
        third_moms,
        n,
        production_constant,
        elasticities,
        labour,
        number_of_levels,
        saving_rates,
        number_of_agents,
    )
    expects = expectations_temp_exp_general(c_moms, temperature, number_of_levels)
    Z = np.dot(n, expects)
    t_rates = np.zeros((number_of_levels, number_of_levels))
    for k in range(number_of_levels):
        for l in range(number_of_levels):
            t_rates[k][l] = (1 - exploration) * n[k] * n[l] * expects[l] / (
                Z * tau
            ) + exploration * n[k] / (tau * number_of_levels)
    return t_rates


def skewness(moments):
    means = moments[0]
    st_devs = np.sqrt(moments[1])
    third_moms = moments[2]

    skews = np.zeros(len(means))
    for i, mean in enumerate(means):
        skews[i] = st_devs[i] ** -3 * (
            third_moms[i] - 3 * means[i] * st_devs[i] ** 2 - 2 * means[i] ** 3
        )
    return skews


# @numba.njit
# def drift_term(
#     t,
#     y,
#     number_of_levels,
#     number_of_agents,
#     production_constant,
#     elasticities,
#     labour,
#     saving_rates,
#     update_time,
#     depreciation,
#     transition_vectors,
#     temperature,
#     exploration,
# ):
#     # Splitting the moments and occupation numbers and applying the exponential, because we are integrating the log
#     log_n = y[-number_of_levels:]
#     n = np.exp(log_n)
#
#     means = y[:number_of_levels]
#     variances = np.exp(y[number_of_levels : 2 * number_of_levels])
#
#     capital_3rd_moments = y[2 * number_of_levels : 3 * number_of_levels]
#     moms = np.array([means, variances, capital_3rd_moments])
#     N = number_of_agents
#     delta = depreciation
#     s = saving_rates
#     L = labour
#     roi = return_on_investment(means, production_constant, elasticities, labour, n)
#     w = wages(n, means, production_constant, elasticities, labour)
#     second_mom = variances + means * means
#     ################################################
#     rates = transition_rates(
#         means,
#         variances,
#         capital_3rd_moments,
#         n,
#         update_time,
#         exploration,
#         number_of_levels,
#         production_constant,
#         elasticities,
#         labour,
#         saving_rates,
#         N,
#         temperature,
#     )
#     ################################################################################################################
#     # dynamical equations for the occupation numbers
#     d_log_n = np.zeros(number_of_levels)
#     for l in range(number_of_levels):
#         for j in range(number_of_levels):
#             d_log_n[l] += (1 / n[l]) * (rates[j][l] - rates[l][j])
#
#     ###################################################################################################################
#     # dynamical equations for the means and third moments
#     d_means = np.zeros(number_of_levels)
#     d_vars = np.zeros(number_of_levels)
#     d_third_moms = np.zeros(number_of_levels)
#     for l in range(number_of_levels):
#         d_means[l] = (roi * s[l] - delta) * means[l] + w * s[l] * L / N
#         d_vars[l] = 2 * (roi * s[l] - delta) * variances[l]
#         d_third_moms[l] = (
#             3 * (roi * s[l] - delta) * capital_3rd_moments[l]
#             + 3 * w * s[l] * L * (variances[l] + means[l] ** 2) / N
#         )
#
#     for l in range(number_of_levels):
#         for i in range(number_of_levels):
#             d_means[l] += ((means[i] - means[l]) / n[l]) * rates[i][l]
#             d_third_moms[l] += (
#                 (capital_3rd_moments[i] - capital_3rd_moments[l]) / n[l]
#             ) * rates[i][l]
#             d_vars[l] += ((second_mom[i] - second_mom[l]) / n[l]) * rates[i][l]
#             d_vars[l] += -2 * means[l] * ((means[i] - means[l]) / n[l]) * rates[i][l]
#             # d_vars[l] += ((means[i] - means[l])/n[l]) ** 2 * rates[i][l] / variances[l] / N
#             # d_vars[l] += -(
#             #                  (second_mom[i] - second_mom[l]) / n[l] - 2 * means[l] * (means[i] - means[l])/n[l]
#             #          ) ** 2 * rates[i][l] / variances[l] ** 2
#     return d_means
#     d_log_vars = d_vars / variances
#     print(t)
#     result = np.concatenate((d_means, d_log_vars, d_third_moms, d_log_n))
#     return result


@numba.njit(cache=True)
def diffusion_matrix(
    _,
    y,
    number_of_levels,
    number_of_agents,
    production_constant,
    elasticities,
    labour,
    saving_rates,
    update_time,
    depreciation,
    transition_vectors,
    temperature,
    exploration,
    gamma,
):
    # Splitting the moments and occupation numbers and applying the exponential, because we are integrating the log
    log_n = y[-number_of_levels:]
    n = np.exp(log_n)

    means = y[:number_of_levels]
    variances = np.exp(y[number_of_levels : 2 * number_of_levels])
    ###############################################
    lower_bound = 10**-20
    for i in numba.prange(number_of_levels):
        if n[i] < lower_bound:
            n[i] = lower_bound
        if variances[i] < lower_bound:
            variances[i] = lower_bound

    ################################################
    capital_3rd_moments = y[2 * number_of_levels : 3 * number_of_levels]
    N = number_of_agents
    second_mom = variances + means * means
    ##################################################
    c_moms = moments_of_consumption_distribution(
        means,
        variances,
        capital_3rd_moments,
        n,
        production_constant,
        elasticities,
        labour,
        number_of_levels,
        saving_rates,
        N,
    )
    ##################################################
    expects = expectations_temp_exp_general(c_moms, temperature, number_of_levels)
    Z = np.dot(n, expects)
    rates = np.zeros((number_of_levels, number_of_levels))
    for k in numba.prange(number_of_levels):
        for l in numba.prange(number_of_levels):
            rates[k][l] = (1 - exploration) * n[k] * n[l] * expects[l] / (
                Z * update_time
            ) + exploration * n[k] / (update_time * number_of_levels)

    ##############################################################################################################
    log_n_diff = np.zeros((number_of_levels, number_of_levels**2))
    means_diff = np.zeros((number_of_levels, number_of_levels**2))
    vars_diff = np.zeros((number_of_levels, number_of_levels**2))
    third_moms_diff = np.zeros((number_of_levels, number_of_levels**2))
    counter = 0
    for i in numba.prange(number_of_levels):
        for j in numba.prange(number_of_levels):
            for l in numba.prange(number_of_levels):
                log_n_diff[l][counter] = (
                    math.sqrt(rates[i][j] / N) / n[l] * transition_vectors[i][j][l]
                )
                ####################################################################################################
                if i == l:
                    means_diff[l][counter] = (
                        (means[j] - means[l])
                        / n[l]
                        / math.sqrt(N)
                        * math.sqrt(rates[j][l])
                    )
                    ####################################################################################################
                    third_moms_diff[l][counter] = (
                        (capital_3rd_moments[j] - capital_3rd_moments[l])
                        / n[l]
                        / math.sqrt(N)
                        * math.sqrt(rates[j][l])
                    )
                    vars_diff[l][counter] = (
                        (
                            (second_mom[j] - second_mom[l]) / n[l]
                            - 2 * means[l] * (means[j] - means[l]) / n[l]
                        )
                        / variances[l]
                        * rates[j][l]
                        / math.sqrt(N)
                    )
            counter += 1
    result = np.concatenate((means_diff, vars_diff, third_moms_diff, log_n_diff))
    return gamma * result


@numba.njit(cache=True)
def drift_beta(
    _,
    y,
    number_of_levels,
    number_of_agents,
    production_constant,
    elasticities,
    labour,
    saving_rates,
    update_time,
    depreciation,
    transition_vectors,
    temperature,
    exploration,
):

    # Splitting the moments and occupation numbers and applying the exponential, because we are integrating the log
    log_n = y[-number_of_levels:]
    n = np.exp(log_n)

    means = y[:number_of_levels]
    variances = np.exp(y[number_of_levels : 2 * number_of_levels])
    second_moments = variances + means * means
    capital_3rd_moments = y[2 * number_of_levels : 3 * number_of_levels]
    ###############################################
    lower_bound = 10**-20
    for i in range(number_of_levels):
        if n[i] < lower_bound:
            n[i] = lower_bound

    ################################################

    con_mom = moments_of_consumption_distribution(
        means,
        variances,
        capital_3rd_moments,
        n,
        production_constant,
        elasticities,
        labour,
        number_of_levels,
        saving_rates,
        number_of_agents,
    )  # uses list[nth-moment][level]

    w = wages(n, means, production_constant, elasticities, labour)
    roi = return_on_investment(means, production_constant, elasticities, labour, n)
    s = saving_rates
    L = labour
    N = number_of_agents
    tau = update_time
    delta = depreciation
    ################################################################################################################
    # dynamical equations for the occupation numbers
    expects = expectations_temp_exp_general(con_mom, temperature, number_of_levels)
    partition_func = np.sum(n * expects)
    dn = np.zeros(number_of_levels)
    dn_explor = np.zeros(number_of_levels)
    for i in range(number_of_levels):
        for j in range(number_of_levels):
            dn += n[i] * n[j] * transition_vectors[i][j] * expects[j]
            dn_explor += n[i] * transition_vectors[i][j]
    dn = 1 / (tau * partition_func) * dn
    dn_explor = 1 / (tau * number_of_levels) * dn_explor
    # Here we combine the exploration and immitating behaviour and divide by n for the change of log(n)
    dn_total = (1 - exploration) * dn + exploration * dn_explor

    d_log_n = dn_total / n
    # This are the dynamical equations for the moments!

    delta_means = np.zeros(number_of_levels)
    delta_vars = np.zeros(number_of_levels)
    delta_3rd_moments = np.zeros(number_of_levels)

    for l in range(number_of_levels):
        delta_means[l] = (roi * s[l] - delta) * means[l] + w * s[l] * L / N
        delta_vars[l] = 2 * (roi * s[l] - delta) * variances[l]
        delta_3rd_moments[l] = (
            3 * (roi * s[l] - delta) * capital_3rd_moments[l]
            + 3 * w * s[l] * L * (variances[l] + means[l] ** 2) / N
        )
    ################################################################################################################
    # adding the jump corrections for the first moment
    individual_mean_jump_corrections = np.zeros((number_of_levels, number_of_levels))
    ind_mean_j_corr_explore = np.zeros((number_of_levels, number_of_levels))
    individual_var_jump_corrections = np.zeros((number_of_levels, number_of_levels))
    ind_var_j_corr_explore = np.zeros((number_of_levels, number_of_levels))
    individual_3rd_mom_jump_corrections = np.zeros((number_of_levels, number_of_levels))
    ind_3rd_mom_j_corr_explore = np.zeros((number_of_levels, number_of_levels))
    # we have to do this three times, due to numba:
    for l in range(number_of_levels):
        for k in range(number_of_levels):
            individual_mean_jump_corrections[l][k] = (
                (means[k] - means[l]) * n[k] * expects[l]
            )
            ind_mean_j_corr_explore[l][k] = (means[k] - means[l]) * n[k]
    # here we technically only add the jump correction for the second non-central moments,
    # going to the variance will only yield an extra term, that will be added later
    for l in range(number_of_levels):
        for k in range(number_of_levels):
            individual_var_jump_corrections[l][k] = (
                (second_moments[k] - second_moments[l]) * n[k] * expects[l]
            )
            ind_var_j_corr_explore[l][k] = (second_moments[k] - second_moments[l]) * n[
                k
            ]
    for l in range(number_of_levels):
        for k in range(number_of_levels):
            individual_3rd_mom_jump_corrections[l][k] = (
                (capital_3rd_moments[k] - capital_3rd_moments[l]) * n[k] * expects[l]
            )
            ind_3rd_mom_j_corr_explore[l][k] = (
                capital_3rd_moments[k] - capital_3rd_moments[l]
            ) * n[k]

    individual_mean_jump_corrections = individual_mean_jump_corrections / (
        tau * partition_func
    )
    individual_var_jump_corrections = individual_var_jump_corrections / (
        tau * partition_func
    )
    individual_3rd_mom_jump_corrections = individual_3rd_mom_jump_corrections / (
        tau * partition_func
    )

    ind_mean_j_corr_explore = ind_mean_j_corr_explore / (tau * number_of_levels * n)
    ind_var_j_corr_explore = ind_var_j_corr_explore / (tau * number_of_levels * n)
    ind_3rd_mom_j_corr_explore = ind_3rd_mom_j_corr_explore / (
        tau * number_of_levels * n
    )

    mean_jump_corrections = (1 - exploration) * np.sum(
        individual_mean_jump_corrections, axis=1
    ) + exploration * np.sum(ind_mean_j_corr_explore, axis=1)
    var_jump_corrections = (1 - exploration) * np.sum(
        individual_var_jump_corrections, axis=1
    ) + exploration * np.sum(ind_var_j_corr_explore, axis=1)
    capital_3rd_moment_jump_corrections = (1 - exploration) * np.sum(
        individual_3rd_mom_jump_corrections, axis=1
    ) + exploration * np.sum(ind_3rd_mom_j_corr_explore, axis=1)
    #################################################################################################################

    # here we do the extra terms for the variance with ito
    ito_terms_imit = np.zeros((number_of_levels, number_of_levels))
    ito_terms_explor = np.zeros((number_of_levels, number_of_levels))
    ito_terms_2_imit = np.zeros((number_of_levels, number_of_levels))
    ito_terms_2_explor = np.zeros((number_of_levels, number_of_levels))
    for l in range(number_of_levels):
        for k in range(number_of_levels):
            ito_terms_imit[l][k] = (
                -(1 / number_of_agents)
                * ((means[k] - means[l]) / n[l]) ** 2
                * (n[k] * n[l] * expects[l])
            )
            ito_terms_explor[l][k] = (
                -(1 / number_of_agents) * ((means[k] - means[l]) / n[l]) ** 2 * n[k]
            )
            ito_terms_2_imit[l][k] = (
                -(1 / number_of_agents)
                * (
                    (second_moments[k] - second_moments[l]) / n[l]
                    - 2 * means[l] * ((means[k] - means[l]) / n[l])
                )
                ** 2
                * (n[k] * n[l] * expects[l])
            )
            ito_terms_2_explor[l][k] = (
                -(1 / number_of_agents)
                * (
                    (second_moments[k] - second_moments[l]) / n[l]
                    - 2 * means[l] * ((means[k] - means[l]) / n[l])
                )
                ** 2
                * n[k]
            )
    ito_terms_imit = ito_terms_imit / (tau * partition_func)
    ito_terms_2_imit = ito_terms_2_imit / (tau * partition_func * variances)
    ito_terms_explor = ito_terms_explor / (tau * number_of_levels)
    ito_terms_2_explor = ito_terms_2_explor / (tau * number_of_levels * variances)
    ito_terms = (1 - exploration) * np.sum(
        ito_terms_imit, axis=1
    ) + exploration * np.sum(ito_terms_explor, axis=1)
    ito_terms_2 = (1 - exploration) * np.sum(
        ito_terms_2_imit, axis=1
    ) + exploration * np.sum(ito_terms_2_explor, axis=1)
    ################################################################################################################
    # her we add the additional term for the variance
    var_jump_corrections += -2 * means * mean_jump_corrections  # + ito_terms
    var_jump_corrections += ito_terms + ito_terms_2
    delta_means += mean_jump_corrections
    # return delta_means
    delta_vars += var_jump_corrections
    delta_3rd_moments += capital_3rd_moment_jump_corrections
    ################################################################################################################
    # putting moments and occupation numbers together
    delta_log_vars = np.zeros(number_of_levels)

    for level in range(number_of_levels):
        if variances[level] > 10**-11:
            delta_log_vars[level] = delta_vars[level] / variances[level]
        else:
            delta_log_vars[level] = 0
    delta_y = np.concatenate((delta_means, delta_log_vars, delta_3rd_moments, d_log_n))
    return delta_y
