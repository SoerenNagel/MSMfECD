import os
import numpy as np
import warnings
import jax
#diffrax gives a future warnig from jax, which is incredably annoying
warnings.simplefilter(action='ignore', category=FutureWarning)
import jax.numpy as jnp
from micro_model import MicroModel
from asano_model import AsanoModel
import matplotlib.pyplot as plt

def main():
    # Set Up of the parameters
    N:int = 500
    Labour = N
    Tau:float = 300.0
    Deprecaition:float = 0.05
    Exploration:float = 0.05
    Error:float= 0.05
    Beta: float = 20.0
    M:int = 5
    Possible_Rates = jnp.linspace(0.05,0.95, M)
    Seed: int= 1234
    Time = 2000
    Steps:int = 2*np.ceil(N*Time/Tau) # number is twice the number of expected transitions

    occupation_nums = [1, 470, 1, 3, 25]
    capital_values = [3, 10, 100, 500, 1200]
    # Create identical inital conditions for both models
    key = jax.random.PRNGKey(Seed)
    saving_rates = jnp.concatenate([Possible_Rates[i]*jnp.ones(n) for i,n in enumerate(occupation_nums)])
    capital = jnp.concatenate([jax.random.uniform(key, shape=(n,), minval=k-1, maxval=k+1)
        for n,k in zip(occupation_nums, capital_values)])

    #data_dir = os.path.join("..", "data")
    # ##########################################################################
    # Simulate Asano model
    asano_model= AsanoModel(
            N=N,
            tau=Tau,
            d_rate=Deprecaition,
            labour= Labour,
            error=Error)
    # set initial data
    asano_model.initialize(jnp.concatenate([saving_rates, capital]))
    #simulate
    asano_model.simulate(stopping_time=Time, steps=Steps, seed=Seed)
 
    # ##########################################################################
    # Simulate our model
    model = MicroModel(
            N=N,
            tau=Tau,
            d_rate=Deprecaition,
            labour= Labour,
            beta=Beta,
            exploration_rate=Exploration)

    # set initial data
    model.initialize(jnp.concatenate([saving_rates, capital]))
    #simulate
    model.simulate(stopping_time=Time, steps=Steps, seed=Seed)

   # ##########################################################################
    # Visualization of the two models
    fig,axs = plt.subplots(1,2)
    # Final saving rate distribution in the second row
    axs[0].set_title("Saving  Rate Distribution")
    model.plot_saving_rate_distribution(ax=axs[0])
    axs[1].set_title("Saving Rate Distribution Asano et.al.")
    asano_model.plot_saving_rate_distribution(ax=axs[1])
    fig.tight_layout()
    plt.savefig("second_sate.py")


if __name__=="__main__":
    main()
