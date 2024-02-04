import jax
import jax.numpy as jnp
import haiku as hk
from typing import Sequence, Callable, Optional, Tuple

class DropoutLayer(hk.Module):
    def __init__(self, input_size: int, init_fn: Optional[Callable]=None):
        super().__init__("dropout")
        self.input_size = input_size
        self.init_fn = init_fn

    def __call__(self, x: jnp.ndarray):
        if self.init_fn is None:
            stddev = 1. / jnp.sqrt(self.input_size)
            self.init_fn = hk.initializers.TruncatedNormal(stddev=stddev)
        w = hk.get_parameter("w", [self.input_size], jnp.float32, init=self.init_fn)
        return x*w


def make_bnn_model(layer_dims: Sequence[int], act_fn: Optional[Callable]=None,
                   init_fn: Optional[Callable]=None, invsp_noise_std: float=1.0,
                   dropout_layer: bool=False):
    """
    Initialise the Bayesian neural network
    :param hidden_sizes: The number of hidden units in each layer.
    :param act_fn: The activation function
    :param init_fn: The initialisation function
    :param invsp_noise_std: The inverse of the noise standard deviation of the output
    :param dropout: Whether to use dropout layer
    """

    if act_fn is None:
        act_fn = jax.nn.relu

    def forward(batch, is_training=True):
        x, _ = batch
        if dropout_layer:
            x = DropoutLayer(x.shape[-1])(x)
        for i, layer_dim in enumerate(layer_dims):
            x = hk.Linear(layer_dim, name=f"linear_{i}",
                          w_init=init_fn, b_init=init_fn)(x)
            x = act_fn(x)
        x = hk.Linear(1)(x)
        x = jnp.concatenate([x, jnp.ones_like(x) * invsp_noise_std], -1)
        return x

    return forward