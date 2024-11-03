import test
from exoemulator.train.opacity import OptExoJAX
from exoemulator.model.mlp import EmuMlp
from exoemulator.model.decoder import EmuMlpDecoder

from flax import nnx
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tqdm
import optax

import orbax.checkpoint as ocp

ckpt_dir = ocp.test_utils.erase_and_create_empty("/home/kawahara/checkpoints")


def test_training():
    from exojax.test.emulate_mdb import mock_mdbExomol
    from exojax.test.emulate_mdb import mock_wavenumber_grid
    from exojax.spec.opacalc import OpaPremodit
    from jax import config

    config.update("jax_enable_x64", True)

    trange = [500.0, 1000.0]
    prange = [1.0e-5, 1.0e2]

    nu_grid, wav, res = mock_wavenumber_grid()
    print(nu_grid.shape)  # 20000
    mdb = mock_mdbExomol()
    opa = OpaPremodit(mdb, nu_grid, auto_trange=trange)
    # emulator_model = EmuMlp(rngs=nnx.Rngs(0), grid_length=len(nu_grid))
    emulator_model = EmuMlpDecoder(rngs=nnx.Rngs(0), grid_length=len(nu_grid))
    opt = OptExoJAX(opa=opa)

    
    # defines metrics
    metrics = nnx.MultiMetric(loss=nnx.metrics.Average("loss"))
    nsample = 100  # default 300
    #niter = 1000
    
    lossarr = []
    testlossarr = []

    # optimizer
    niter = 1000000
    momentum = 0.9
    learning_rate_arr = np.logspace(-4, -6, 5)
    N_lr = len(learning_rate_arr)

    tag = "decoder_"+str(N_lr)+"lr"
    outfile = "mlp_emulator_" + tag + ".png"
    print("outfile:", outfile)


    for j in range(N_lr):
        learning_rate = learning_rate_arr[j] #learning rate scheduling (step decay)
        optimizer = nnx.Optimizer(emulator_model, optax.adamw(learning_rate, momentum))
        description = "learning rate: "+str(learning_rate)+":"+str(j+1)+"/"+str(N_lr)
        for i in tqdm.tqdm(range(niter),desc=description):
            if np.mod(i, 20) == 0:
                input_parameters, logxs = opt.generate_batch(
                    trange=trange, prange=prange, nsample=nsample, method="lhs"
                )
            loss = opt.train_step(
                emulator_model, optimizer, metrics, input_parameters, logxs
            )
            if np.mod(i, 100) == 0:
                input_parameters, logxs = opt.generate_batch(
                    trange=trange, prange=prange, nsample=nsample, method="lhs"
                )
                testloss = opt.evaluate_step(emulator_model, input_parameters, logxs)
                lossarr.append(loss)
                testlossarr.append(testloss)
        
        
    
    

    # plot loss
    np.savez("loss"+tag+".npz", lossarr=lossarr, testlossarr=testlossarr)

    # test prediction
    input_par = jnp.array([729.0, jnp.log10(3.0e-1)])
    output_vector = emulator_model(input_par)
    xs_ref = opt.opa.xsvector(input_par[0], 10 ** input_par[1])
    offset = 22.0
    factor = 0.3
    xs_pred = xs_prediction(output_vector, offset, factor)

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
    plt.savefig(outfile)  # R: lerning rate 1e-4
    plt.show()

    _, state = nnx.split(emulator_model)
    nnx.display(state)
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(ckpt_dir / "state", state)

    # need to wait for the file to be written
    from exoemulator.utils.sleep import wait_for_saving

    wait_for_saving(waitsec=3)


def xs_prediction(output_vector, offset, factor):
    return 10 ** (output_vector / factor - offset)


if __name__ == "__main__":
    test_training()
