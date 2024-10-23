from exoemulator.utils.checkpackage import check_installed
from exoemulator.train.opacity import OptExoJAX
from exoemulator.model.mlp import EmuMlp
import numpy as np
from flax import nnx


def test_OptExoJAX():
    opt = OptExoJAX()


def test_OptExoJAX_opa_premodit():
    if check_installed("exojax"):
        from exojax.test.emulate_mdb import mock_mdbExomol
        from exojax.test.emulate_mdb import mock_wavenumber_grid
        from exojax.spec.opacalc import OpaPremodit
        from jax import config

        config.update("jax_enable_x64", True)
        nu_grid, wav, res = mock_wavenumber_grid()
        mdb = mock_mdbExomol()
        opa = OpaPremodit(mdb, nu_grid, auto_trange=[490.0, 510.0])
        opt = OptExoJAX(opa=opa)
        nsample = 10
        input_parameters, xs = opt.generate_batch(
            trange=[490.0, 510.0], prange=[1.0e-3, 1.0e3], nsample=nsample, method="lhs"
        )
        input_parameters2, xs2 = opt.generate_batch(
            trange=[490.0, 510.0], prange=[1.0e-3, 1.0e3], nsample=nsample, method="lhs"
        )

        assert np.all(
            input_parameters != input_parameters2
        ) and input_parameters.shape == (
            nsample,
            2,
        )
        assert np.all(xs != xs2) and xs.shape == (nsample, len(nu_grid))

    else:
        pass


if __name__ == "__main__":
    test_OptExoJAX()
    test_OptExoJAX_opa_premodit()
