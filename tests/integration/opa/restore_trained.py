"""restoring the trained model and predicting the cross section
"""

import matplotlib.pyplot as plt
from jax import numpy as jnp
from exoemulator.model.decoder import opaemulator_decoder
from exoemulator.train.opacity import OptExoJAX
from pathlib import Path


#ckpt_dir = Path("/home/kawahara/checkpoints")
ckpt_dir = Path("/home/kawahara/checkpoints_tmp")
#checkpointer = ocp.StandardCheckpointer()


opt = OptExoJAX()
emulator_model = opt.restore_model(opaemulator_decoder, ckpt_dir)
input_par = jnp.array([729.0, jnp.log10(3.0e-1)])
output_vector = emulator_model(input_par)
xs_pred = opt.xs_prediction(output_vector)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(opt.nu_grid, xs_pred, alpha=1, lw=2)
plt.yscale("log")
plt.savefig("restored.png")  # R: lerning rate 1e-4


