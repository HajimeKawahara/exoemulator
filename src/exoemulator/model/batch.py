def generate_batches(x, y, batch_size):
    """generate batches

    Args:
        x (jnp.array): x
        y (jnp.array): y
        batch_size (int): batch size

    Returns:
        generator
    """
    n = x.shape[0]
    for i in range(0, n, batch_size):
        yield x[i:i + batch_size], y[i:i + batch_size]

