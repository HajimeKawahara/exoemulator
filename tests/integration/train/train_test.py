from exoemulator.train.opacity import OptExoJAX
from exoemulator.model.mlp import EmuMlp
from flax import nnx
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tqdm
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
    emulator_model = EmuMlp(rngs=nnx.Rngs(0), grid_length=len(nu_grid))

    opt = OptExoJAX(opa=opa)

    # optimizer
    learning_rate = 1e-3
    momentum = 0.9
    optimizer = nnx.Optimizer(emulator_model, optax.adamw(learning_rate, momentum))

    # defines metrics
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))

    nsample = 300

    for i in tqdm.tqdm(range(300)):
        input_parameters, xs = opt.generate_batch(
            trange=trange, prange=prange, nsample=nsample, method="lhs"
        )
        opt.train_step(emulator_model, optimizer, metrics, input_parameters, xs)

    input_par = jnp.array([729.0, jnp.log10(3.0e-1)])
    output_vector = emulator_model(input_par)
    xs_ref = opt.opa.xsvector(input_par[0], 10 ** input_par[1])

    fig = plt.figure()
    ax = fig.add_subplot(211)
    plt.plot(nu_grid, output_vector, alpha=0.3, lw=2)
    plt.plot(nu_grid, np.log10(xs_ref), lw=1)
    ax2 = fig.add_subplot(212)
    plt.plot(nu_grid, np.log10(xs_ref) - output_vector, lw=1)
    ax2.set_xlabel("wavenumber (cm-1)")
    plt.show()


if __name__ == "__main__":
    test_training()
