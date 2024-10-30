import jax
import jax.numpy as jnp
from jax import random
from flax import nnx
import tqdm
import optax
import time


class EmuMlpDecoder(nnx.Module):
    """simple decoder type MLP neuralnet emulator model

    Args:
        nnx (_type_): nnx module

    """

    def __init__(self, *, rngs: nnx.Rngs, grid_length: int):
        #self.Nneuron_last=1024
        self.Nneuron_last = 512
        self.dense_entrance = nnx.Linear(in_features=2, out_features=16, rngs=rngs)
        self.dense_1 = nnx.Linear(in_features=16, out_features=32, rngs=rngs)
        self.dense_2 = nnx.Linear(in_features=32, out_features=256, rngs=rngs)
        self.dense_3 = nnx.Linear(in_features=256, out_features=self.Nneuron_last, rngs=rngs)

        self.dense_out = nnx.Linear(
            in_features=self.Nneuron_last, out_features=grid_length, rngs=rngs
        )

    def __call__(self, input_parameter):
        x = nnx.gelu(self.dense_entrance(input_parameter))
        x = nnx.gelu(self.dense_1(x))
        x = nnx.gelu(self.dense_2(x))
        x = nnx.gelu(self.dense_3(x))
        return self.dense_out(x)
        # return nnx.sigmoid(self.dense_out(x)) #limit to [0,1]


class EmuUpsDecoder(nnx.Module):
    """simple upsampling decoder type neuralnet emulator model (not working yet)

    Args:
        nnx (_type_): nnx module

    """

    # assuming grid_length = 20000
    def __init__(self, *, rngs: nnx.Rngs, grid_length: int):
        self.dense_entrance = nnx.Linear(in_features=2, out_features=16, rngs=rngs)
        self.dense_1 = nnx.Linear(in_features=16, out_features=32, rngs=rngs)
        self.dense_2 = nnx.Linear(in_features=32, out_features=256, rngs=rngs)
        self.dense_3 = nnx.Linear(in_features=256, out_features=1000, rngs=rngs)

        # input should have the length = 5000
        exansion = 20
        self.transconv_out = nnx.ConvTranspose(
            in_features=1,
            out_features=1,
            strides=exansion,
            kernel_size=(1,),
            padding="SAME",
            rngs=rngs,
        )

    def __call__(self, input_parameter):
        x = nnx.gelu(self.dense_entrance(input_parameter))
        x = nnx.gelu(self.dense_1(x))
        x = nnx.gelu(self.dense_2(x))
        x = nnx.gelu(self.dense_3(x))
        x = self.transconv_out(x[:, :, jnp.newaxis])
        return x[:, :, 0]
        # return nnx.sigmoid(self.dense_out(x)) #limit to [0,1]
