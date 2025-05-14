from datetime import datetime
import warnings
import os
warnings.simplefilter(action='ignore', category=FutureWarning)
from typing import Union, Tuple
from jaxtyping import Float, Array, Int, jaxtyped
from typeguard import typechecked as typechecker
import numpy as np
import jax
import jax.numpy as jnp
import tqdm
import seaborn as sns
import logging
from datetime import datetime
# Configure the logger
logging.basicConfig(
    filename='../logs/error_log.log',
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt="%Y-%m-%d %H:%M:%S"  # Specify date format
)
import json

import state_functions as sf
from integration import integrator, update_saving_rates


def calculate_temporal_grid(intervals:Float[Array, "num_jumps+2 2"],
                            num_steps:int,
                            final_time: float|int) -> Tuple:
    dt = final_time/num_steps
    steps = jnp.zeros(len(intervals),dtype=jnp.int32)
    for i, (t0, t1) in enumerate(intervals):
        if t1-t0 < dt:
            steps = steps.at[i].set(1)
            continue
        steps = steps.at[i].set(int(jnp.ceil((t1-t0)/dt)))

    # we have the number of steps, with an additional one added by the integrator on the first interval
    total_step_number = jnp.sum(steps)+1
    indexes = jnp.cumsum(steps)
    indexes = indexes + 1 # account for the extra step that will be added by the integrator we shift all of them by 1
    indexes = jnp.insert(indexes, 0, 0) #insert the first index
    indexes = indexes.at[-1].add(1) # for the stopping time
    index_pairs = [(indexes[i], indexes[i+1]) for i in range(len(intervals))]
    return total_step_number, steps, index_pairs


class MicroModel:
    """
    This class implements the micromodel for our paper.
    """
    def __init__(self, 
                 N: int,
                 tau: int| float,
                 d_rate:float,
                 labour: Union[int, float],
                 beta:Union[int, float],
                 exploration_rate: Union[int, float],
                 ) -> None:

        assert isinstance(N, int), "The number of agnets 'N'  has to be an intgerer"
        assert N > 0 , "The number of agnets needs to be a positive and non-zero"
        self.N:int = N

        assert isinstance(tau, float) or isinstance(tau, int), "The rate tau has to be a float, or int"
        assert tau > 0 , "The rate tau needs to be a positive and non-zero"
        self.tau :float= float(tau)

        assert isinstance(d_rate, float) or isinstance(d_rate, int), "The depreciation rate has to be a float, or int"
        assert d_rate > 0 , "The depreciation rate needs to be a positive and non-zero"
        self.d_rate:float = float(d_rate)

        assert isinstance(beta, float) or isinstance(beta, int), "The temperature has to be a float, or int"
        assert beta > 0 , "The innverse temperaature needs to be a positive and non-zero"
        self.beta:float = float(beta)

        assert isinstance(labour, float) or isinstance(labour, int), "The rate tau has to be a float, or int"
        assert labour > 0 , "Labour needs to be a positive and non-zero"
        self.labour:float= float(labour)

        self.num_levels :int = 0 # Number of availablesaving rates

        assert isinstance(exploration_rate, float)
        assert (0 < exploration_rate < 1), "Exploration rate need to be in the interval (0,1)"
        self.exploaration_rate:float= exploration_rate

        self.solution : None | Float[Array, "steps, {2*self.N}"]  = None
        self.time :None | Float[Array, "steps"]   = None

        self.available_saving_rates: tuple|None=None


    def __repr__(self) -> str:
        return f'MicroModel(N={self.N}, tau={self.tau}, kappa={self.d_rate},beta={self.beta}, L={self.labour}, M={self.num_levels}, epsilon={self.exploaration_rate})'

    def initialize(self, initial_data: Float[Array,"{2*self.N}"]):
        """Note that the available saving rates are set by the initial_data.
        In the macroscopic model the levels can not die out anyway.
        The initial_data takes shape [saving_rates, capitals]"""
        assert isinstance(initial_data, jnp.ndarray)
        assert initial_data.shape == (2*self.N,), "'Initial_data' has to be of shape (2*N,)"
        assert all(initial_data[-self.N:] >= 0) , "Capital needs to be non-negative"
        assert np.all(
            (0 < initial_data[:self.N]) & (initial_data[:self.N] < 1)), "saving rates need to be in the interval (0,1)"

        self.solution = jnp.array([initial_data])
        self.num_levels = jnp.unique(initial_data[:self.N]).size

        self.available_saving_rates = tuple(jnp.unique(self.get_saving_rates()).tolist())


    def get_saving_rates(self, timestep:int|None=None) -> Float[Array, "{self.N}"]:
        """ returns the saving rates of the solution for a given timestep, or for the last time stepbydefault.""" 
        assert self.solution is not None, "Model has not been simulated"
        if  jnp.isnan(jnp.sum(self.solution[:,:self.N])):
            warnings.warn("'nan' found in saving rates")

        if timestep is None:
            return self.solution[-1][:self.N]
        else:
            return self.solution[timestep][:self.N]

    def get_saving_rate_series(self) -> Float[Array, "{len(self.time)} {self.N}"]:
        assert self.solution is not None, "Model has not been simulated"
        return self.solution[:, :self.N]

    def get_capital(self, timestep=None) -> Float[Array, "{self.N}"]:
        """ returns the saving rates of the solution for a given timestep, or for the last time stepbydefault.""" 
        assert self.solution is not None, "Model has not been simulated"
        if  jnp.isnan(jnp.sum(self.solution[:][-self.N:])):
            warnings.warn("'nan' found in saving rates")
        if timestep is None:
            return self.solution[-1][-self.N:]
        else:
            return self.solution[timestep][-self.N:]

    def get_capital_series(self) -> Float[Array, "{len(self.time)} {self.N}"]:
        assert self.solution is not None, "Model has not been simulated"
        return self.solution[:, -self.N:]

    def get_poisson_intervals(self, final_time:float) -> Float[Array, "num_jumps+2 2"]:
        """returns the time intervals for integratio. I_k = [t_k, t_(k+1)]"""
        inter_arrival_times = np.random.exponential(self.tau/self.N, # mean arrival time tau/N
                                                    int(2*final_time/self.tau*self.N)
                                                    )
        arrival_times = np.cumsum(inter_arrival_times)

        # make sure, we only have switching times before the final time
        arrival_times = arrival_times[arrival_times<final_time] 
        arrival_times = np.insert(arrival_times, 0, 0)
        arrival_times = np.append(arrival_times, final_time)

        intervals = jnp.array(
            [jnp.array([arrival_times[i],arrival_times[i+1]]) for i in range(len(arrival_times) -1 )]
        )

        return intervals

    @jaxtyped(typechecker=typechecker)
    def update_saving_rates(self, agent:Int[Array, ""],
                            saving_rates:Float[Array, "{self.N}"],
                            seed:Int[Array, ""],
                            capital: Float[Array, "{self.N}"]
                            )-> Float[Array, "{self.N}"]:
        assert self.available_saving_rates is not None, "Model was not initialized"
        return update_saving_rates(agent=agent,
                                   saving_rates=saving_rates,
                                   available_saving_rates=self.available_saving_rates,
                                   seed = seed,
                                   capital=capital,
                                   labour=jnp.array(self.labour,
                                                    dtype=jnp.float32),
                                   beta = jnp.array(self.beta, dtype=jnp.float32),
                                   exploration =jnp.array(self.exploaration_rate,
                                                          dtype=jnp.float32)
                                  )
    def simulate_manual(self, stopping_time:float|int, steps: int,  seed:int)-> None:
        # inital checks
        assert self.solution is not None, "Mode has to be initialized!"
        assert len(self.solution) == 1, "Model has already been simulated"
        assert stopping_time > 0, "'stopping_time' has to be non-zero"

        # generate the Poissson process
        intervals = self.get_poisson_intervals(stopping_time)
        #sample the agnets that switch, the two extra are just for the followinng loop
        switching_agents = jnp.array(
            np.random.choice([i for i in range(self.N)], size=len(intervals))
        , dtype=jnp.int32)

        #generate seeds for all the samples
        key = jax.random.PRNGKey(seed)
        seeds = jax.random.randint(key, minval=0, maxval=10**9, shape=(len(intervals),))

        #get initial conditions
        saving_rates = self.get_saving_rates()
        capital = self.get_capital()

        # main integration loop
        for i, interval in tqdm.tqdm(enumerate(intervals)):
            dt = interval[1]-interval[0]

            if dt==0:
                warnings.warn("dt=0")
                dt = 10**-5
            capital = capital + sf.capital_evolution(capital=capital, saving_rates=saving_rates, labour= jnp.array(self.labour), depreciation_rate=jnp.array(self.d_rate)) * dt
            if jnp.isnan(jnp.sum(capital)):
                warnings.warn("Solver returned 'nan'")
            elif jnp.min(capital) < 0:
                raise(RuntimeError("The solver returned negatice capital"))

            #Update the saving rates
            saving_rates = self.update_saving_rates(agent=switching_agents[i],
                                                    saving_rates=saving_rates,
                                                    seed = seeds[i],
                                                    capital=capital,
                                                    )
            # update initial capital

            assert jnp.min(saving_rates) >=0
            assert jnp.max(saving_rates) <=1

        self.solution = jnp.array([jnp.concatenate([saving_rates, capital])])
        self.time = jnp.array([intervals[-1][1]])

    def simulate(self, stopping_time:float|int, steps: int,  seed:int)-> None:
        # inital checks
        assert self.solution is not None, "Mode has to be initialized!"
        assert len(self.solution) == 1, "Model has already been simulated"
        assert stopping_time > 0, "'stopping_time' has to be non-zero"
        #assert isinstance(save_all, bool), "'save_all' needs to be a bool"


        # generate the Poissson process
        intervals = self.get_poisson_intervals(stopping_time)
        #sample the agnets that switch, the two extra are just for the followinng loop
        switching_agents = jnp.array(
            np.random.choice([i for i in range(self.N)], size=len(intervals))
        , dtype=jnp.int32)

        #get the time discretization
        num_steps, stepping, index_pairs = calculate_temporal_grid(intervals, steps, stopping_time)

        #generate seeds for all the samples
        key = jax.random.PRNGKey(seed)
        seeds = jax.random.randint(key, minval=0, maxval=10**9, shape=(len(intervals),))

        #get initial conditions
        saving_rates = self.get_saving_rates()
        initial_capital = self.get_capital()

        #pre allocation
        self.solution = jnp.zeros((int(num_steps), 2*self.N))
        self.time = jnp.zeros(int(num_steps))

        print(f'The smallest time innterval is {min([interval[1]-interval[0] for interval in intervals])} and occurs at interval {jnp.argmin(jnp.array([interval[1]-interval[0] for interval in intervals]))}')

        # main integration loop
        for i, interval in tqdm.tqdm(enumerate(intervals)):
            ts = jnp.linspace(interval[0],
                              interval[1],
                              stepping[i],
                              endpoint=False
                            )
            try:
                times, capitals = integrator(time_interval=jnp.array(interval),
                                         saving_rates=saving_rates,
                                         labour=jnp.array(self.labour),
                                         depreciation_rate=jnp.array(self.d_rate),
                                         initial_capital = initial_capital,
                                         ts=ts,
                                         save_t_end=True if i==0 else False,
                                         )
            except Exception as e:
                data={
                      "Time Interval": interval.tolist(),
                      "Saving Rates": saving_rates.tolist(),
                      "Labour":  float(self.labour),
                      "Depreccaition": float(self.d_rate),
                      "Initial_Capital": initial_capital.tolist(),
                }
                formatted_data = json.dumps(data, indent=4)
                logging.error("'Max Steps' reached while integrating with data:\n%s", formatted_data, exc_info=True)
                raise e

            if jnp.isnan(jnp.sum(capitals[-1])):
                warnings.warn("Solver returned 'nan'")
            elif jnp.min(capitals[-1]) < 0:
                raise(RuntimeError("The solver returned negatice capital"))

            #Save the solution
            saving_rate_time_series = jnp.tile(saving_rates, (len(capitals),1))
            ind0, ind1 = index_pairs[i]
            self.solution = self.solution.at[ind0:ind1].set(jnp.hstack([saving_rate_time_series,capitals]))
            self.time = self.time.at[ind0:ind1].set(times)

            #Update the saving rates
            saving_rates = self.update_saving_rates(agent=switching_agents[i],
                                                    saving_rates=saving_rates,
                                                    seed = seeds[i],
                                                    capital=capitals[-1],
                                                    )
            # update initial capital
            initial_capital = capitals[-1]

            assert jnp.min(saving_rates) >=0
            assert jnp.max(saving_rates) <=1

    def save(self, data_dir):
        # Generate a base filename from __repr__()
        file_base = os.path.join(data_dir, self.__repr__())

        # Check if file already exists and append a timestamp if it does
        file = file_base
        if os.path.isfile(file + ".npz"):
            now = datetime.now()
            timestamp = now.strftime("%Y%m%d_%H%M%S")  # Format safe for filenames
            file = f"{file_base}_{timestamp}"

        # Save data
        np.savez(file + ".npz", time=np.array(self.time), solution=np.array(self.solution))

    def clear(self):
        assert self.solution is not None
        self. solution = jnp.array([self.solution[0]])

    def load(self, data_dir, file_name: str | None = None):
        # Set file path depending on whether a file name was provided
        file = os.path.join(data_dir, file_name or self.__repr__() + ".npz")

        # Load data
        with np.load(file) as data:
            self.time = data['time']
            self.solution = data['solution']

    def plot_saving_rates(self, ax):
        assert self.solution is not None, "Model has not been intialized"
        ax.plot(self.time,self.get_saving_rate_series(), color="grey",  lw=.8)
        ax.plot(self.time, jnp.mean(self.get_saving_rate_series(),axis=1),
                color="tab:red",
                label=r'Mean Saving')
        return ax

    def plot_saving_rate_distribution(self,ax,time_step:int|None=None):
        assert self.solution is not None, "Model has not been intialized"
        saving_rates= np.array(self.get_saving_rates(time_step))
        bins= np.linspace(-0.001,1.001, 21)
        sns.histplot(data=saving_rates, kde=False, ax=ax, bins=bins)

        ax.set_yscale('log')
        return ax

    def plot_production(self, ax):
        assert self.solution is not None, "Model has not been intialized"
        capitals = self.get_capital_series()
        aggeragate_capital = jax.vmap(sf.aggregate_capital)(capitals)

        production = jax.vmap(
            sf.production, in_axes=[0, None]
        )(aggeragate_capital, jnp.array(self.labour))
        if jnp.isnan(jnp.sum(production)):
            warnings.warn("'nan' occured in prosuction")
        ax.plot(self.time, production)
        return ax

    def plot_capital_distribution(self,ax, time_step:int|None=None):
        assert self.solution is not None, "Model has not been intialized"
        capital = np.array(self.get_capital(time_step))
        sns.histplot(data=capital, kde=True, ax=ax)
        return ax
