""" example of training a simple MLP model using flax
"""

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import random
from flax import nnx
import tqdm
import optax
import time


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


class MLP(nnx.Module):
    """simple MLP model Like the Payne

    Args:
        nnx (_type_): nnx module

    """

    def __init__(self, *, rngs: nnx.Rngs):
        grid_length = 1024
        nneuron = 128
        self.dense_entrance = nnx.Linear(in_features=1, out_features=nneuron, rngs=rngs)
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


def loss_fn(model: MLP, input_parameter, label_vector):
    output_vector = model(input_parameter)
    loss = jnp.mean((output_vector - label_vector) ** 2)
    return loss, output_vector


@nnx.jit
def train_step(
    model: MLP,
    optimizer: nnx.Optimizer,
    metric: nnx.MultiMetric,
    input_parameter,
    label_vector,
):
    grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
    (loss, output_vector), grads = grad_fn(model, input_parameter, label_vector)
    metric.update(loss=loss)
    optimizer.update(grads)


@nnx.jit
def eval_step(
    model: MLP,
    metric: nnx.MultiMetric,
    input_parameter,
    label_vector,
):
    batch_output_vector = model(input_parameter)
    loss = jnp.mean((batch_output_vector - label_vector) ** 2)
    metric.update(loss=loss)


if __name__ == "__main__":
    import jax.numpy as jnp
    import numpy as np
    import optax
    from exoemulator.model.batch import generate_batches

    grid_length = 1024
    nsample = 512
    input_phase = np.random.rand(nsample) * 2 * np.pi
    _x, label_sin_vector = generate_sin_curve(input_phase, 0.0, grid_length)
    input_phase = input_phase.reshape(-1, 1)
    print(input_phase.shape, label_sin_vector.shape)

    # batch
    batch_size = 32
    input_parameter_minibatches = generate_batches(input_phase, batch_size)
    label_vector_minibatches = generate_batches(label_sin_vector, batch_size)

    # MLP model
    model = MLP(rngs=nnx.Rngs(0))
    nnx.display(model)
    
    # single input parameter
    output_vector = model(input_phase[0])
    print(input_phase[0].shape, "->", output_vector.shape)
    # batch input parameter
    output_vector = model(input_parameter_minibatches[0])
    print(input_parameter_minibatches[0].shape, "->", output_vector.shape)

    # optimizer
    learning_rate = 1e-3
    momentum = 0.9
    optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, momentum))

    # defines metrics
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))

    # training
    batch_training = True
    # batch_training= False

    start = time.time()
    if batch_training:
        print("batch training")
        nepoch = 1000
    else:
        print("minibatch training")
        nepoch = 100
    
    for iepoch in tqdm.trange(nepoch):
        #training
        if batch_training:
            train_step(model, optimizer, metrics, input_phase, label_sin_vector)
        else:
            for istep, minibatch_input_parameter in enumerate(
                input_parameter_minibatches
            ):
                minibatch_label_vector = label_vector_minibatches[istep]
                train_step(
                    model,
                    optimizer,
                    metrics,
                    minibatch_input_parameter,
                    minibatch_label_vector,
                )
        # evaluation
        

    print("elapsed time:", time.time() - start)
    # single input parameter
    phase = jnp.pi
    input_phase = jnp.ones((1,)) * phase
    output_vector = model(input_phase)

    fig = plt.figure()
    ax = fig.add_subplot(211)
    plt.plot(_x, output_vector, "-")
    plt.plot(_x, jnp.sin(_x + phase), alpha=0.3, lw=4)

    ax = fig.add_subplot(212)
    plt.plot(_x, output_vector - jnp.sin(_x + phase))

    plt.savefig("sin.png")
    plt.show()
