def latin_hypercube_sampling(xrange, yrange, n: int):
    from scipy.stats.qmc import LatinHypercube

    # Create a latin hypercube sampler
    lhc = LatinHypercube(d=2)
    # Sample the latin hypercube
    samples = lhc.random(n)
    dx = xrange[1] - xrange[0]
    dy = yrange[1] - yrange[0]
    samples[:, 0] = samples[:, 0] * dx + xrange[0]
    samples[:, 1] = samples[:, 1] * dy + yrange[0]
    return samples
