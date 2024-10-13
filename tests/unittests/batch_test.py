from exoemulator.model.batch import generate_batches

import jax.numpy as jnp


def test_generate_batches():
    x = jnp.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = jnp.array([1, 2, 3, 4])
    batch_size = 2

    batches = list(generate_batches(x, y, batch_size))

    assert len(batches) == 2
    assert (batches[0][0] == jnp.array([[1, 2], [3, 4]])).all()
    assert (batches[0][1] == jnp.array([1, 2])).all()
    assert (batches[1][0] == jnp.array([[5, 6], [7, 8]])).all()
    assert (batches[1][1] == jnp.array([3, 4])).all()


def test_generate_batches_incomplete_batch():
    x = jnp.array([[1, 2], [3, 4], [5, 6]])
    y = jnp.array([1, 2, 3])
    batch_size = 2

    batches = list(generate_batches(x, y, batch_size))

    assert len(batches) == 2
    assert (batches[0][0] == jnp.array([[1, 2], [3, 4]])).all()
    assert (batches[0][1] == jnp.array([1, 2])).all()
    assert (batches[1][0] == jnp.array([[5, 6]])).all()
    assert (batches[1][1] == jnp.array([3])).all()

def test_generate_batches_temp():
    data = jnp.array([1,2,3,4,5,6,7,8,9,10])
    batch_size = 4
    batches = list(generate_batches(data, data, batch_size))
    print(batches)

if __name__ == "__main__":
    test_generate_batches()
    test_generate_batches_incomplete_batch()
    test_generate_batches_temp()