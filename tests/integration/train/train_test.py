from exoemulator.train.opacity import OptExoJAX
from exoemulator.model.mlp import NemMlp
from flax import nnx
import numpy as np


def test_training():
    from exojax.test.emulate_mdb import mock_mdbExomol
    from exojax.test.emulate_mdb import mock_wavenumber_grid
    from exojax.spec.opacalc import OpaPremodit
    from jax import config

    config.update("jax_enable_x64", True)
    
    trange = [500.0, 1000.0]
    prange=[1.0e-5, 1.0e2]

    nu_grid, wav, res = mock_wavenumber_grid()
    mdb = mock_mdbExomol()
    opa = OpaPremodit(mdb, nu_grid, auto_trange=trange)
    opt = OptExoJAX(opa=opa)
    nem = NemMlp(rngs=nnx.Rngs(0))
    
    nsample = 100
    
    tarr, parr, xs = opt.generate_batch(
        trange=trange, prange=prange, nsample=nsample, method="lhs"
    )
    opt.train()

if __name__ == "__main__":
    test_training()
