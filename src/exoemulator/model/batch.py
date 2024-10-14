import jax.numpy as jnp


def generate_batches(x, batch_size: int):
    """generate batches as a list of device arrays

    Args:
        x (np.array): x
        batch_size (int): batch size

    Returns:
        list of device arrays (jnp.array)

    """
    n = x.shape[0]
    device_arrays = []

    for i in range(0, n, batch_size):
        device_arrays.append(jnp.asarray(x[i : i + batch_size]))

    return device_arrays
