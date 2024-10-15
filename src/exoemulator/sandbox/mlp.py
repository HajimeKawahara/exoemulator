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


def loss_fn(model: MlpLikePayne, batch_input_parameter, batch_label_vector):
    batch_output_vector = model(batch_input_parameter)
    loss = jnp.mean((batch_output_vector - batch_label_vector) ** 2)
    return loss, batch_output_vector


@nnx.jit
def train_step(
    model: MlpLikePayne,
    optimizer: nnx.Optimizer,
    metric: nnx.MultiMetric,
    batch_input_parameter,
    batch_label_vector,
):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, batch_output_vector), grads = grad_fn(
        model, batch_input_parameter, batch_label_vector
    )
    #metric.update(loss=loss)
    optimizer.update(grads)


@nnx.jit
def eval_step(
    model: MlpLikePayne,
    metric: nnx.MultiMetric,
    batch_input_parameter,
    batch_label_vector,
):
    batch_output_vector = model(batch_input_parameter)
    loss = jnp.mean((batch_output_vector - batch_label_vector) ** 2)
    #metric.update(loss=loss)


if __name__ == "__main__":
    import jax.numpy as jnp
    import numpy as np
    import optax

    grid_length = 1024
    nlabel = 100
    input_phase = np.random.rand(nlabel) * 2 * np.pi

    _x, label_sin_vector = generate_sin_curve(input_phase, 0.0, grid_length)
    input_phase = input_phase.reshape(-1, 1)
    print(input_phase.shape, label_sin_vector.shape)

    model = MlpLikePayne(rngs=nnx.Rngs(0))

    # single input parameter
    input_parameter = jnp.ones((1,))
    output_vector = model(input_parameter)
    print(input_parameter.shape, "->", output_vector.shape)

    # batch input parameter
    nbatch = 64
    batch_input_parameter = jnp.ones((nbatch, 1))
    output_vector = model(batch_input_parameter)
    print(batch_input_parameter.shape, "->", output_vector.shape)

    # optimizer
    learning_rate = 1e-3
    momentum = 0.9
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, momentum))
    metrics = nnx.MultiMetric(
        accuracy=nnx.metrics.Accuracy(), loss=nnx.metrics.Average("loss")
    )

    # training
    no_batch = True
    if no_batch:
        # nonbatch training 
        for i in tqdm.trange(1000):
            train_step(model, optimizer, metrics, input_phase, label_sin_vector)

        # single input parameter
        input_parameter = jnp.ones((1,))*jnp.pi
        output_vector = model(input_parameter)
        plt.plot(_x, output_vector, "-")
        plt.savefig("sin.png")
        plt.show()  
    else:
        print("")
