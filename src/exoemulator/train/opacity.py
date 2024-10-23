"""opacity model training (opt).

    - spectrum training (spt)

"""

from exoemulator.utils.checkpackage import check_installed
from exoemulator.train.sampling import latin_hypercube_sampling
import numpy as np


class OptExoJAX:
    """Opacity Training with ExoJAX"""

    def __init__(self, opa=None, emu=None):
        check_installed("exojax")
        self.set_opa(opa)
        self.set_emulator(emu)

    def set_opa(self, opa):
        """sets opa (opacity class in ExoJAX)

        Args:
            opa : "opa" (opacity class) in ExoJAX

        Notes:
            opa is an opacity class in ExoJAX, OpaPremodit, OpaModit, or OpaDirect

        """
        if opa is None:
            print("opa (ExoJAX opacity class) has not been given yet")
        else:
            print("opa (ExoJAX opacity class) in opt: ", opa.__class__.__name__)
            self.opa = opa

    def set_emulator(self, emu):
        """sets emulator model

        Args:
            emu : emulator model

        """
        if emu is None:
            print("emu (emulator model) has not been given yet")
        else:
            print("emu (emulator model) in opt: ", emu.__class__.__name__)
            self.emu = emu

    def generate_batch(self, trange, prange, nsample, method="lhs"):
        if method == "lhs":
            # tarr is linear, parr is log
            samples = latin_hypercube_sampling(trange, np.log10(prange), nsample)
            tarr = samples[:, 0]
            parr = 10 ** (samples[:, 1])
        else:
            raise NotImplementedError(f"Method {method} is not implemented")
        return tarr, parr, self.opa.xsmatrix(tarr, parr)


if __name__ == "__main__":
    opt = OptExoJAX()
