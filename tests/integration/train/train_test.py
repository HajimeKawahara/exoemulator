from exoemulator.train.opacity import OptExoJAX
from exoemulator.model.mlp import EmuMlp
from flax import nnx
import numpy as np
import optax

def test_training():
    from exojax.test.emulate_mdb import mock_mdbExomol
    from exojax.test.emulate_mdb import mock_wavenumber_grid
    from exojax.spec.opacalc import OpaPremodit
    from jax import config

    config.update("jax_enable_x64", True)

    trange = [500.0, 1000.0]
    prange = [1.0e-5, 1.0e2]

    nu_grid, wav, res = mock_wavenumber_grid()
    mdb = mock_mdbExomol()
    opa = OpaPremodit(mdb, nu_grid, auto_trange=trange)
    emu = EmuMlp(rngs=nnx.Rngs(0))

    opt = OptExoJAX(opa=opa, emu=emu)

    # optimizer
    learning_rate = 1e-3
    momentum = 0.9
    optimizer = nnx.Optimizer(emu, optax.adamw(learning_rate, momentum))

    # defines metrics
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))

    nsample = 100

    input_parameters, xs = opt.generate_batch(
        trange=trange, prange=prange, nsample=nsample, method="lhs"
    )
    opt.train_step(optimizer, metrics, input_parameters, xs)


if __name__ == "__main__":
    test_training()
