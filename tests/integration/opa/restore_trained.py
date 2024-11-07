"""restoring the trained model and predicting the cross section
"""

import matplotlib.pyplot as plt
from exoemulator.model.decoder import opaemulator_decoder
from exoemulator.model.opa import OpaEmulator
from pathlib import Path


ckpt_dir = Path("/home/kawahara/checkpoints")
#ckpt_dir = Path("/home/kawahara/checkpoints_tmp")

opa = OpaEmulator(opaemulator_decoder, ckpt_dir)
xs_pred = opa.xsvector(729.0, 3.0e-1)


fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(opa.nu_grid, xs_pred, alpha=1, lw=2)
plt.yscale("log")
plt.savefig("restored.png")


