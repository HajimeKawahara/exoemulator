""" example of training a simple MLP model using flax
"""

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import random
from flax import nnx
import tqdm
import optax


def generate_sin_curve(phase, std, num_points):
    """generates a sin curve with some noise

    Args:
        phase (_type_): phase shift of the sin curve
        std (_type_): standard deviation of the noise
        num_points (_type_): number of points in the curve

    Returns:
        x, y=sin(x+phase)+noise: x and y values of the curve
    """
    x = jnp.linspace(0, 4 * jnp.pi, num_points)
    y = jnp.sin(x + phase[:, np.newaxis]) + std * random.normal(
        random.PRNGKey(0), (len(phase), num_points)
    )
    return x, y


def plot_curve(x, y, ypred=None):
    """plots the curve

    Args:
        x (_type_): x
        y (_type_): y
        ypred (_type_, optional): y prediction. Defaults to None.

    Returns:
        plt: plot object
    """
    plt.plot(x, y, ".", alpha=0.5)
    if ypred is not None:
        plt.plot(x, ypred, "-")
    plt.xlabel("x")
    plt.ylabel("y")
    return plt


class MlpLikePayne(nnx.Module):
    """simple MLP model Like the Payne

    Args:
        nnx (_type_): nnx module

    """

    def __init__(self, *, rngs: nnx.Rngs):
        grid_length = 1024
        self.dense_entrance = nnx.Linear(in_features=1, out_features=128, rngs=rngs)
        self.dense = nnx.Linear(in_features=128, out_features=128, rngs=rngs)
        self.dense_out = nnx.Linear(
            in_features=128, out_features=grid_length, rngs=rngs
        )

    def __call__(self, input_parameter):
        x = nnx.gelu(self.dense_entrance(input_parameter))
        x = nnx.gelu(self.dense(x))
        x = nnx.gelu(self.dense(x))
        return self.dense_out(x)
        # return nnx.sigmoid(self.dense_out(x)) #limit to [0,1]


def loss_fn(model: MlpLikePayne, batch):
    output_vector = model(batch)


@jax.jit
def train_step(state, x, y):
    """update function

    Args:
        state (_type_): state of the model
        x (_type_): x
        y (_type_): y

    Returns:
        updated state of the model
    """
    loss, grad = jax.value_and_grad(loss_fn, argnums=1)(state, state.params, x, y)
    updates, new_opt_state = state.tx.update(grad, state.opt_state)
    new_params = optax.apply_updates(state.params, updates)
    return state.replace(params=new_params, opt_state=new_opt_state), loss


if __name__ == "__main__":
    import jax.numpy as jnp
    import numpy as np
    import optax

    grid_length = 1024
    nlabel = 100
    phase_label = np.random.rand(nlabel) * 2 * np.pi

    x, y = generate_sin_curve(phase_label, 0.0, grid_length)
    # y = y.reshape(-1, 1)
    # x = x.reshape(-1, 1)
    phase_label = phase_label.reshape(-1, 1)
    print(phase_label.shape, x.shape, y.shape)

    model = MlpLikePayne(rngs=nnx.Rngs(0))

    # single input parameter
    input_parameter = jnp.ones((1,))
    y = model(input_parameter)
    print(input_parameter.shape, "->", y.shape)

    # batch input parameter
    nbatch = 64
    batch_input_parameter = jnp.ones((nbatch, 1))
    y = model(batch_input_parameter)
    print(batch_input_parameter.shape, "->", y.shape)

    exit()

    learning_rate = 1e-3
    momentum = 0.9
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, momentum))
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(), loss=nnx.metrics.Average("loss")
    )
