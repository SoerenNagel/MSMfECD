import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import jax
from  jaxtyping import Array, Float, Int, jaxtyped
from jax import lax
from typeguard import typechecked as typechecker
#jax.config.update("jax_enable_x64", True) #uncomment this to enable float64 on the GPU
import jax.numpy as jnp
from functools import partial
from state_functions import capital_evolution, transition_probabilities, consumption
from diffrax import (
    diffeqsolve,
    ODETerm,
    SaveAt,
    Euler,
)


@jaxtyped(typechecker=typechecker)
@partial(
    jax.jit,
    static_argnames=[
        "save_t_end",
    ],
)
def integrator(
    time_interval:Float[Array,"2"],
    saving_rates: Float[Array, "N"],
    labour:Float[Array,""],
    depreciation_rate:Float[Array,""],
    initial_capital: Float[Array, "N"],
    ts:Float[Array, "steps"],
    save_t_end:bool
) -> tuple:
    # set up the drift and diffusion terms
    drift = lambda _t, y, _args: capital_evolution(y,
                                                   jnp.array(labour,dtype=jnp.float32),
                                                   saving_rates,
                                                   depreciation_rate
                                                   )

    # specify when the solution will be saved
    save_at = SaveAt(ts=ts, t1=save_t_end)

    # specify the solver
    solver = Euler()

    dt = jnp.diff(time_interval)[0]
    # run the solver
    sol = diffeqsolve(
        ODETerm(drift),
        solver,
        t0=time_interval[0],
        t1=time_interval[1],
        dt0=jnp.max(jnp.array([dt*0.8, 10**-5])),
        y0=initial_capital,
        saveat=save_at,
        max_steps=10**7
    )
    return sol.ts, sol.ys

@jaxtyped(typechecker=typechecker)
#@partial(jax.jit, static_argnames=["available_saving_rates"])
def update_saving_rates(
        agent:Int[Array, ""],
        saving_rates:Float[Array, "N"],
        available_saving_rates: tuple,
        seed:Int[Array, ""],
        capital: Float[Array, "N"],
        labour:Float[Array, ""],
        beta:Float[Array, ""],
        exploration: Float[Array, ""]
)->Float[Array, "N"]:

    """This function updates the saving rate of an agent according to the transition
    probailities. The agents to adapt can be sampled in advance."""
    key = jax.random.PRNGKey(seed)
    new_saving_rate = jax.random.choice(key=key,
                                        a=jnp.array(available_saving_rates),
                                        p=transition_probabilities(capital,available_saving_rates,
                                        saving_rates,
labour, beta, exploration))
    saving_rates = saving_rates.at[agent].set(new_saving_rate)
    return saving_rates

@jaxtyped(typechecker=typechecker)
def update_saving_rates_asano(
        agent:Int[Array, ""],
        saving_rates:Float[Array, "N"],
        seed:Int[Array, ""],
        capital: Float[Array, "N"],
        labour:Float[Array, ""],
        error:Float[Array, ""],
        )->jnp.ndarray:
    """This function updates the saving rate of an agent according to the transition
    probailities. The agents to adapt can be sampled in advance."""
    key = jax.random.PRNGKey(seed)

    consumptions = consumption(capital,saving_rates, labour)
    s = saving_rates[jnp.argmax(consumptions)] + jax.random.uniform(key, minval=-error, maxval=error)
    new_saving_rate = lax.cond(0 <= s <= 1,                             # if s in [0,1] 
                    lambda s: s, # return s
                               lambda s: lax.cond(s > 1, # elif s> 1
                                       lambda s: jnp.float32(1), #return 1
                                       lambda s: jnp.float32(0), # else retrun 0
                                       s), 
                    s)
    saving_rates = saving_rates.at[agent].set(new_saving_rate)
    return saving_rates



