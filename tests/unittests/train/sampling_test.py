import numpy as np
from exoemulator.train.sampling import latin_hypercube_sampling


def test_sample_shape():
    xrange = (0, 10)
    yrange = (0, 10)
    n = 5
    samples = latin_hypercube_sampling(xrange, yrange, n)
    assert samples.shape == (n, 2)


def test_sample_range():
    xrange = (0, 10)
    yrange = (0, 10)
    n = 5
    samples = latin_hypercube_sampling(xrange, yrange, n)
    assert np.all(samples[:, 0] >= xrange[0]) and np.all(samples[:, 0] <= xrange[1])
    assert np.all(samples[:, 1] >= yrange[0]) and np.all(samples[:, 1] <= yrange[1])


def test_different_samples():
    xrange = (0, 10)
    yrange = (0, 10)
    n = 5
    samples1 = latin_hypercube_sampling(xrange, yrange, n)
    samples2 = latin_hypercube_sampling(xrange, yrange, n)
    assert not np.array_equal(samples1, samples2)
