import jax
import jax.numpy as jnp
from jax import random
from flax import nnx
import tqdm
import optax
import time


class EmuMlp(nnx.Module):
    """simple MLP neuralnet emulator model

    Args:
        nnx (_type_): nnx module

    """

    def __init__(self, *, rngs: nnx.Rngs, grid_length: int):
        nneuron = 1024
        self.dense_entrance = nnx.Linear(in_features=2, out_features=nneuron, rngs=rngs)
        self.dense = nnx.Linear(in_features=nneuron, out_features=nneuron, rngs=rngs)
        self.dense_out = nnx.Linear(
            in_features=nneuron, out_features=grid_length, rngs=rngs
        )

    def __call__(self, input_parameter):
        x = nnx.gelu(self.dense_entrance(input_parameter))
        x = nnx.gelu(self.dense(x))
        x = nnx.gelu(self.dense(x))
        x = nnx.gelu(self.dense(x))
        x = nnx.gelu(self.dense(x))
        return self.dense_out(x)
        # return nnx.sigmoid(self.dense_out(x)) #limit to [0,1]


