from exoemulator.model.batch import generate_batches

import jax.numpy as jnp


def test_generate_batches_single():
    import numpy as np

    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    batch_size = 4
    batches = generate_batches(data, batch_size)
    print(batches)

    assert (batches[0] == jnp.array([1, 2, 3, 4])).all()
    assert (batches[1] == jnp.array([5, 6, 7, 8])).all()
    assert (batches[2] == jnp.array([9, 10])).all()


if __name__ == "__main__":
    test_generate_batches_single()
