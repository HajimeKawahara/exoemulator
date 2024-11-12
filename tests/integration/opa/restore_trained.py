"""restoring the trained model and predicting the cross section
"""
import jax.numpy as jnp
import matplotlib.pyplot as plt
from exoemulator.model.decoder import opaemulator_decoder
from exoemulator.model.opa import OpaEmulator
from pathlib import Path

ckpt_dir = Path("/home/kawahara/ckp/schedule4")

# cross section vector
opa = OpaEmulator(opaemulator_decoder, ckpt_dir)
xs_pred = opa.xsvector(729.0, 3.0e-1)


# cross section matrix
Tarr = jnp.array([729.0, 1000.0])
Parr = jnp.array([3.0e-1, 3.0e1])
xs_mat = opa.xsmatrix(Tarr, Parr)

print(xs_mat.shape)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(opa.nu_grid, xs_pred, alpha=1, lw=2)
plt.plot(opa.nu_grid, xs_mat[0,:], alpha=1, lw=2, ls="dashed")
plt.plot(opa.nu_grid, xs_mat[1,:], alpha=1, lw=2, ls="dashed")

plt.yscale("log")
plt.savefig("restored.png")



exit(0)