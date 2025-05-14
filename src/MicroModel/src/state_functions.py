import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import jax
import jax.numpy as jnp
from functools import partial
from jaxtyping import Float, Array, jaxtyped
from typeguard import typechecked as typechecker

@jaxtyped(typechecker=typechecker)
@jax.jit
def aggregate_capital(capital: Float[Array, "N"]) -> Float[Array,""]:
    """returns the aggregate capita, i.e. the sum of all capital stocks"""
    return jnp.sum(capital)

@jaxtyped(typechecker=typechecker)
@jax.jit
def production(aggregate_capital:Float[Array,""],
               labour:Float[Array,""],
               )->Float[Array,""]:
    """returns the economic production for a production constant of 1 and elsticities
    of 0.5"""
    return jnp.sqrt(aggregate_capital*labour)

@jaxtyped(typechecker=typechecker)
@jax.jit
def returns(capitals:Float[Array, "N"], labour:Float[Array,""]) -> Float[Array,""]:
    return jnp.sqrt(labour/aggregate_capital(capitals))/2

@jaxtyped(typechecker=typechecker)
def wage(capitals:Float[Array, "N"], labour:Float[Array,""])->Float[Array,""]:
    return jnp.sqrt(aggregate_capital(capitals)/labour)/2

@jaxtyped(typechecker=typechecker)
@jax.jit
def income(capital:Float[Array, "N"],
           labour:Float[Array,""])->Float[Array, "N"]:
    r = returns(capital, labour)
    w = wage(capital, labour)
    return r*capital + w*labour/len(capital)

@jaxtyped(typechecker=typechecker)
@jax.jit
def capital_evolution(capital: Float[Array,"N"],
                      labour:Float[Array, ""],
                      saving_rates: Float[Array,"N"],
                      depreciation_rate:Float[Array, ""])->Float[Array, "N"]:
    """The evolution equation for the capital stocks"""
    return  income(capital,labour)*saving_rates - depreciation_rate*capital 

@jaxtyped(typechecker=typechecker)
@jax.jit
def consumption(capital:Float[Array, "N"],
                saving_rates: Float[Array, "N"],
                labour:Float[Array,""]) -> Float[Array, "N"]:
    return (1-saving_rates) * income(capital,labour)

@jaxtyped(typechecker=typechecker)
#@partial(jax.jit, static_argnames=["available_saving_rates"])
def transition_probabilities(capital:Float[Array, "N"],
                             available_saving_rates: tuple,
                             saving_rates:Float[Array, "N"],
                             labour: Float[Array,""],
                             beta:Float[Array, ""],
                             exploration:Float[Array, ""],
                             )->Float[Array, "M"]:
    consumptions= beta*consumption(capital, saving_rates,  labour)
    # the softmax is invariant under a uniform shift. Like this we avoid overflow in the exponential
    consumptions = consumptions - jnp.max(consumptions)
    probabilities = jnp.array([
        jnp.sum(jnp.exp(jnp.array(jnp.where(saving_rates == s, consumptions, -jnp.inf))))
        for s in available_saving_rates
    ])

    # Propper normalization
    probabilities = probabilities/jnp.sum(probabilities)

    #Add the exploration probabilities
    probabilities = (1-exploration) * probabilities + exploration/len(available_saving_rates)
    if jnp.isnan(jnp.sum(probabilities)):
        warnings.warn("'Nan' in Probabilities")
    return probabilities

