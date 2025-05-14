import jax.numpy as jnp
from jaxtyping import Float, Array, Int, jaxtyped
from typeguard import typechecked as typechecker
from typing import Union
from integration import update_saving_rates_asano
from micro_model import MicroModel


class AsanoModel(MicroModel):
    """
    This class implements the micromodel for our paper.
    """
    def __init__(self, 
                 N: int,
                 tau: Union[int, float],
                 d_rate:Union[int, float],
                 labour: Union[int, float],
                 error: Union[int, float],
                 ) -> None:

        super().__init__(N, tau, d_rate, labour,jnp.inf, error)
        # set the unused arguments to None
        self.saving_rates = None
        self.beta = 10**10 #should not be used
        self.error = self.exploaration_rate

    def __repr__(self) -> str:
        return f'AsanoModel(N={self.N}, tau={self.tau}, kappa={self.d_rate}, L={self.labour}, error={self.exploaration_rate})'

    @jaxtyped(typechecker=typechecker)
    def update_saving_rates(self, agent:Int[Array, ""],
                            saving_rates:Float[Array, "{self.N}"],
                            seed:Int[Array, ""],
                            capital: Float[Array, "{self.N}"]
                            )-> Float[Array, "{self.N}"]:
        return update_saving_rates_asano(agent=agent,
                                           saving_rates=saving_rates,
                                           seed = seed,
                                           capital=capital,
                                         labour=jnp.array(self.labour),
                                           error = jnp.array(self.error),
                                           )
