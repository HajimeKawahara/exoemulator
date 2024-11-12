from exoemulator.train.opacity import OptExoJAX
from exoemulator.model.decoder import opaemulator_decoder
import matplotlib.pyplot as plt
from flax import nnx
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

import orbax.checkpoint as ocp

ckpt_dir = ocp.test_utils.erase_and_create_empty("/home/kawahara/ckp/schedule3")
# ckpt_dir = ocp.test_utils.erase_and_create_empty("/home/kawahara/checkpoints_tmp")


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
    opt = OptExoJAX(opa=OpaPremodit(mdb, nu_grid, auto_trange=trange))

    model = opaemulator_decoder(rngs=nnx.Rngs(0), grid_length=len(nu_grid))
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))

    # scheduler
    learning_rate_arr = [1.0e-4, 1.0e-5, 1.0e-6]
    nepoch_arr = [3000000, 3000000, 3000000]
    batch_size_arr = [100, 100, 100] # [100, 100, 100]
    train_update_interval_arr = [20, 20, 20] # [20, 20, 20]
    tag = "decoder_3schedule" # 
    outfile = "mlp_emulator_" + tag + ".png"
    print("outfile:", outfile)

    opt.train(
        model,
        metrics,
        trange,
        prange,
        learning_rate_arr,
        batch_size_arr,
        train_update_interval_arr,
        nepoch_arr,
    )

    # save loss (use plotloss.py for plotting)
    np.savez("loss" + tag + ".npz", lossarr=opt.lossarr, testlossarr=opt.testlossarr)

    # test prediction
    input_par = jnp.array([729.0, jnp.log10(3.0e-1)])
    output_vector = model(input_par)
    xs_ref = opt.opa.xsvector(input_par[0], 10 ** input_par[1])
    xs_pred = opt.xs_prediction(output_vector)

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(211)
    plt.plot(nu_grid, xs_ref, lw=1)
    plt.plot(nu_grid, xs_pred, alpha=0.2, lw=2)
    plt.yscale("log")
    ax.set_ylabel("cross section (cm2)")
    ax2 = fig.add_subplot(212)
    plt.plot(nu_grid, 1.0 - xs_pred / xs_ref, lw=1)
    ax2.set_xlabel("wavenumber (cm-1)")
    ax2.set_ylabel("relative error")
    plt.savefig(outfile)
    #    plt.show()

    _, state = nnx.split(model)
    # nnx.display(state)

    opt.save_state(ckpt_dir, state)


if __name__ == "__main__":
    test_training()
