""" example of training a simple MLP model using flax
"""

import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.training import train_state
import tqdm
import optax


class MLP6(nn.Module):
    """ simple MLP model with 6 layers
    
    Args:
        nn (_type_): linen module

    """
    features: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=self.features)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.features)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.features)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.features)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.features)(x)
        x = nn.relu(x)
        x = nn.Dense(features=1)(x)
        return x


@jax.jit
def loss_fn(state, params, x, y):
    """ loss function
    
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
def update_fn(state, x, y):
    """ update function
    
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
