"""opacity model training (opt).

    - spectrum training (spt)

"""

from exoemulator.utils.checkpackage import check_installed
from exoemulator.train.sampling import latin_hypercube_sampling
import numpy as np

class OptExoJAX:
    """Opacity Training with ExoJAX"""

    def __init__(self, opa=None):
        check_installed("exojax")
        self.set_opa(opa)

    def set_opa(self, opa):
        if opa is None:
            print("opa has not been given yet")
        else:
            print("opa in opt: ", opa.__class__.__name__)
            self.opa = opa

    def generate_batch(self, trange, prange, nsample, method="lhs"):
        if method == "lhs":
            # tarr is linear, parr is log
            samples = latin_hypercube_sampling(trange, np.log10(prange), nsample)
            tarr = samples[:, 0]
            parr = 10**(samples[:, 1])
        else:
            raise NotImplementedError(f"Method {method} is not implemented")
        return tarr, parr, self.opa.xsmatrix(tarr, parr)


if __name__ == "__main__":
    opt = OptExoJAX()
