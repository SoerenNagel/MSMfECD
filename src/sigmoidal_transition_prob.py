import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate

import phase_space_functions as psf
from itertools import product
import statsmodels.api as sm
import plot_libarry as plt_lib
import sdeint


# This class just holds the solution of the social_process so that it has the correct data structure for plotting
class bin_social_process:
    def __init__(
        self, occupation_numbers, initial_occupation_numbers, number_of_levels, times
    ):
        self.initial_occupation_numbers = np.array(initial_occupation_numbers)
        self.time_points = times
        self.occupation_numbers = occupation_numbers
        self.number_of_levels = number_of_levels


class Economic_Process_sigmoidal:
    def __init__(
        self,
        number_of_agents: int,
        number_of_levels: int,
        update_time: int | float,
        temperature: int | float,
        initial_occupation_numbers: np.ndarray,
        exploration: float,
        initial_capital_moments: np.ndarray,
        labour: int | float,
        elasticities: list | np.ndarray = np.array([0.5, 0.5]),
        production_constant: int | float = 1,
        depreciation: float = 0.05,
        gamma: int | float = 1,
    ) -> None:
        #     level_of_leading_consumer)
        # defining data for the social process
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
        self.elasticities = np.array(elasticities, np.float64)
        self.production_constant = production_constant
        self.depreciation = depreciation
        self.relative_noise = gamma
        self.social_process = [0]
        self.times = np.array([0], dtype=np.float64)
        self.capital_moments = [
            np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=np.float64)
        ]

        self.saving_rates = np.linspace(
            0.05, 0.95, self.number_of_levels, dtype=np.float64
        )

        self.initial_capital_moments = np.array(
            initial_capital_moments, dtype=np.float64
        )
        # use data structure, s.t. capital_moments = [[first moments], [second moments], [third moments]]
        self.transition_vectors = np.zeros(
            (self.number_of_levels, self.number_of_levels, self.number_of_levels)
        )
        for i, j in product(range(self.number_of_levels), range(self.number_of_levels)):
            self.transition_vectors[i][j] = np.eye(
                1, self.number_of_levels, j
            ) - np.eye(1, self.number_of_levels, i)
        self.transition_vectors = np.array(self.transition_vectors, dtype=np.float64)
        # for self.transition_vectors[i][j] we get the transition vector for a transition from level i to level j

    def unpacking_occupation_numbers(self) -> np.ndarray:
        t = self.times
        n_bin = np.zeros((1, self.number_of_levels))
        times_for_n = np.zeros(1)
        for sol in self.social_process:
            n_bin = np.concatenate((n_bin, sol.occupation_numbers))
            times_for_n = np.concatenate((times_for_n, sol.time_points))
        n_bin = np.delete(n_bin, 0, axis=0)
        times_for_n = np.delete(times_for_n, 0)
        n_bin = np.transpose(n_bin)
        occupation_numbers = np.zeros((self.number_of_levels, len(t)))
        for l in range(self.number_of_levels):
            occupation_numbers[l] = np.interp(t, times_for_n, n_bin[l])
        return np.transpose(occupation_numbers)

    def dynamical_equations(self, t: float, y: np.ndarray) -> np.ndarray:
        # This is just for unpacking all the parameters rom the class
        # the original function needs to be in phase_space_functions due to the numba usage
        return psf.sigmoidal_dynamical_equations(
            t,
            y,
            self.number_of_levels,
            self.number_of_agents,
            self.production_constant,
            self.elasticities,
            self.labour,
            self.saving_rates,
            self.update_time,
            self.depreciation,
            self.transition_vectors,
            self.temperature,
            self.exploration,
        )

    def solve_economics(self, target_time: int) -> None:
        self.initial_capital_moments[1] = np.log(self.initial_capital_moments[1])
        initial_data = np.concatenate(
            (
                self.initial_capital_moments.reshape(3 * self.number_of_levels),
                np.log(self.initial_occupation_numbers),
            )
        )
        sol = integrate.solve_ivp(
            self.dynamical_equations, [0, target_time], initial_data, method="Radau"
        )

        occupation_numbers = []
        for y in np.transpose(sol.y):
            log_y_moments = y[: self.number_of_levels * 3]
            n = np.exp(y[self.number_of_levels * 3 :])
            log_moments = log_y_moments.reshape((3, self.number_of_levels))
            moments = np.array([log_moments[0], np.exp(log_moments[1]), log_moments[2]])
            self.capital_moments.append(moments)
            occupation_numbers.append(n)
        del self.capital_moments[0]
        self.times = sol.t
        self.social_process[0] = bin_social_process(
            occupation_numbers,
            self.initial_occupation_numbers,
            self.number_of_levels,
            sol.t,
        )

    def correlation_time(self, cut_off_time: float) -> float:
        index = 0
        t = self.times[index]
        while t < cut_off_time:
            index += 1
            t = self.times[index]

        data_time_points = np.array(
            self.times[-len(self.times) + index :], dtype=np.float64
        )

        data = np.zeros(len(data_time_points))
        n = self.unpacking_occupation_numbers()
        for i in range(len(data_time_points)):
            data[i] = psf.production(
                self.capital_moments[i + index][0],
                self.production_constant,
                n[i + index],
                self.elasticities,
                self.labour,
            )
        lags = np.arange(len(data_time_points))
        acorr = sm.tsa.acf(data, nlags=len(lags) - 1)
        #######################################################################
        # correlation time
        corr_time = np.trapz(np.absolute(acorr))
        return corr_time

    def spike_timing_intervals(self):
        occupation_numbers = self.unpacking_occupation_numbers()
        mean_saving_rate = np.zeros(len(self.times))
        for i, n in enumerate(occupation_numbers):
            mean_saving_rate[i] = np.dot(n, self.saving_rates)
        d_ms = np.diff(mean_saving_rate)
        window = 100
        average_d_ms = []

        for ind in range(len(d_ms) + 1 - window):
            average_d_ms.append(np.mean(d_ms[ind : ind + window]))
        for ind in range(window - 1):
            average_d_ms.insert(0, np.nan)
        plt.plot(average_d_ms)
        plt.show()
        threshold = float(input("Threshold = "))
        iter = 0
        spike_timing_intervals = {"low_saving": [], "high_saving": []}
        spike_timings = {"start": [], "end": []}
        while iter <= len(average_d_ms) - 1:
            while np.abs(average_d_ms[iter]) >= threshold:
                iter += 1
            if average_d_ms[iter] < 0:
                state = "low_saving"
            else:
                state = "high_saving"
            interval_start = self.times[iter]
            spike_timings["start"].append(self.times[iter])
            while np.abs(average_d_ms[iter]) < threshold:
                if not (iter == len(average_d_ms) - 1):
                    iter += 1
                else:
                    break
            spike_interval = self.times[iter] - interval_start
            spike_timings["end"].append(self.times[iter])
            spike_timing_intervals[state].append(spike_interval)
            iter += 1

        _, ax = plt.subplots()
        ax.plot(average_d_ms)
        for start in spike_timings["start"]:
            ax.plot([start, start], [0, 1], color="tab:green")
        for end in spike_timings["end"]:
            ax.plot([end, end], [0, 1], color="tab:red")
        plt.show()

    def drift(self, y: np.ndarray, t: float) -> np.ndarray:
        # This is just for unpacking all the parameters rom the class
        # the original function needs to be in phase_space_functions due to the numba usage
        return psf.drift_beta(
            t,
            y,
            self.number_of_levels,
            self.number_of_agents,
            self.production_constant,
            self.elasticities,
            self.labour,
            self.saving_rates,
            self.update_time,
            self.depreciation,
            self.transition_vectors,
            self.temperature,
            self.exploration,
        )

    def diffusion(self, y: np.ndarray, t: float) -> np.ndarray:
        # This is just for unpacking all the parameters rom the class
        # the original function needs to be in phase_space_functions due to the numba usage
        return psf.diffusion_matrix(
            t,
            y,
            self.number_of_levels,
            self.number_of_agents,
            self.production_constant,
            self.elasticities,
            self.labour,
            self.saving_rates,
            self.update_time,
            self.depreciation,
            self.transition_vectors,
            self.temperature,
            self.exploration,
            self.relative_noise,
        )

    def solve_SDE_economics(self, target_time: int, sample_rate: int) -> None:
        """Solves the Stochastic Differential Equations (SDEs) for the economic model.

        This method simulates the time evolution of the system's capital moments
        and occupation numbers up to a `target_time` using Ito integration via
        the `sdeint.itoint` function.

        Instead of integrarting the SDE directly, we intgerate the log of the variances,
        in order to avoid the variances taking negative values.

        The results (time series of capital moments and occupation numbers)
        are stored in `self.capital_moments` and `self.social_process`
        respectively. The simulation time points are stored in `self.times`.
        """
        tspan = np.linspace(0, target_time, sample_rate * target_time)
        self.initial_capital_moments[1] = np.log(self.initial_capital_moments[1])
        initial_data = np.concatenate(
            (
                self.initial_capital_moments.reshape(3 * self.number_of_levels),
                np.log(self.initial_occupation_numbers),
            )
        )

        sol = sdeint.itoint(self.drift, self.diffusion, initial_data, tspan)
        # print('finished integration')
        occupation_numbers = []
        for y in sol:
            log_y_moments = y[: self.number_of_levels * 3]
            n = np.exp(y[self.number_of_levels * 3 :])
            log_moments = log_y_moments.reshape((3, self.number_of_levels))
            moments = np.array([log_moments[0], np.exp(log_moments[1]), log_moments[2]])
            self.capital_moments.append(moments)
            occupation_numbers.append(n)
        del self.capital_moments[0]
        self.times = tspan
        self.social_process[0] = bin_social_process(
            occupation_numbers,
            self.initial_occupation_numbers,
            self.number_of_levels,
            tspan,
        )

    def delete_tail_of_solution(self, num_steps: int) -> None:
        self.capital_moments = self.capital_moments[:-num_steps]
        self.times = self.times[:-num_steps]
        self.social_process[0].occupation_numbers = self.social_process[
            0
        ].occupation_numbers[:-num_steps]
        self.social_process[0].time_points = self.social_process[0].time_points[
            :-num_steps
        ]
        ######################################################

    def continue_solving(self, additional_time: int, sample_rate: int) -> None:
        tspan = np.linspace(
            self.times[-1],
            self.times[-1] + additional_time,
            sample_rate * additional_time,
        )
        initial_capital_moments = self.capital_moments[-1]
        initial_capital_moments[1] = np.log(initial_capital_moments[1])
        initial_occupation_numbers = self.unpacking_occupation_numbers()[-1]
        initial_data = np.concatenate(
            (
                initial_capital_moments.reshape(3 * self.number_of_levels),
                np.log(initial_occupation_numbers),
            )
        )
        sol = sdeint.itoint(self.drift, self.diffusion, initial_data, tspan)
        #####################################################
        # deleting the initial conditions to avoid doubling
        self.capital_moments = self.capital_moments[:-1]
        self.times = self.times[:-1]
        self.social_process[0].occupation_numbers = self.social_process[
            0
        ].occupation_numbers[:-1]
        self.social_process[0].time_points = self.social_process[0].time_points[:-1]
        ######################################################
        for y in sol:
            log_y_moments = y[: self.number_of_levels * 3]
            n = np.exp(y[self.number_of_levels * 3 :])
            log_moments = log_y_moments.reshape((3, self.number_of_levels))
            moments = np.array([log_moments[0], np.exp(log_moments[1]), log_moments[2]])
            self.capital_moments = np.append(self.capital_moments, [moments], axis=0)
            self.social_process[0].occupation_numbers = np.append(
                self.social_process[0].occupation_numbers, [n], axis=0
            )
        self.times = np.concatenate((self.times, tspan))
        self.social_process[0].time_points = self.times

