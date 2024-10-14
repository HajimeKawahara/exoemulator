""" example of training a simple MLP model using flax
"""

import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.training import train_state
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
    y = jnp.sin(x + phase[:,np.newaxis]) + std * random.normal(random.PRNGKey(0), (len(phase), num_points))
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


class MlpLikePayne(nn.Module):
    """simple MLP model Like the Payne

    Args:
        nn (_type_): linen module

    """

    @nn.compact
    def __call__(self, labels):

        grid_length = 1024
        layers = [128, 128]

        _x = labels
        for features in layers:
            _x = nn.Dense(features)(_x)
            _x = nn.gelu(_x)
        _x = nn.Dense(grid_length)(_x)
        return _x
        #return nn.sigmoid(_x) #limit to [0,1]


@jax.jit
def loss_fn(state, params, x, y):
    """loss function

    Args:
        state: state of the model
        params: parameters of the model
        x (jnp.array): x
        y (jnp.array): y

    Returns:
        L2 loss
    """

    y_pred = state.apply_fn(params, x)
    return jnp.mean((y - y_pred) ** 2)


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
    grid_length = 1024
    nlabel= 100
    phase_label = np.random.rand(nlabel)*2*np.pi


    x, y = generate_sin_curve(phase_label, 0.0, grid_length)
    #y = y.reshape(-1, 1)
    #x = x.reshape(-1, 1)
    phase_label = phase_label.reshape(-1, 1)
    print(phase_label.shape, x.shape, y.shape)

    #for i in range(nlabel):
    #    plt.plot(x, y[i, :])
    #plt.show()
    
    # plot_sin_curve(x[:, 0], y[:, 0])

    model = MlpLikePayne()

    rng = random.PRNGKey(0)
#    params = model.init(rng, jnp.array([0.3]))
#    y_pred = model.apply(params, jnp.array([0.3]))

    params = model.init(rng, phase_label)
    y_pred = model.apply(params, phase_label)


    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optax.adam(learning_rate=0.01),
    )

    loss_arr = []
    for _ in tqdm.tqdm(range(20000)):
        state, loss = train_step(state, phase_label, y)
        loss_arr.append(loss)  # loss should decrease

    y_pred = model.apply(state.params, phase_label)
    print(y_pred.shape)
    for i in range(0,2):
        plt.plot(x, y[i,:], ls="solid", color="C"+str(i+1), alpha=0.3)
        plt.plot(x, y_pred[i,:], ls="dashed", color="C"+str(i+1))
    plt.show()

    # validation
    phase_label = np.random.rand(nlabel)*2*np.pi
    x, y = generate_sin_curve(phase_label, 0.0, grid_length)
    phase_label = phase_label.reshape(-1, 1)
    y_pred = model.apply(state.params, phase_label)
    print(y_pred.shape)
    for i in range(0,2):
        plt.plot(x, y[i,:], ls="solid", color="C"+str(i+1), alpha=0.3)
        plt.plot(x, y_pred[i,:], ls="dashed", color="C"+str(i+1))

    plt.show()
    exit()


#    plt = plot_curve(x[:, 0], y[:, 0], y_pred[:, 0])
#    plt.savefig("mlp.png")
