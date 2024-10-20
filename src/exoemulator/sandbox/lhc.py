# Latin hypercube
from scipy.stats.qmc import LatinHypercube

if __name__ == "__main__":
    import jax.numpy as jnp
    import numpy as np
    import matplotlib.pyplot as plt

    # Generate some data
    n = 30
    # Create a latin hypercube sampler
    lhc = LatinHypercube(d=2)

    # Sample the latin hypercube
    samples = lhc.random(n)
    print(samples.shape)
    # Plot the samples
    plt.scatter(samples[:,0], samples[:,1], color="red", label="LHC")
    plt.plot(samples[:,0], np.zeros(n),".", color="blue", label="LHC")
    plt.plot(np.zeros(n), samples[:,1],".", color="blue", label="LHC")
    
    plt.legend()
    plt.savefig("lhc.png")
    plt.show()
