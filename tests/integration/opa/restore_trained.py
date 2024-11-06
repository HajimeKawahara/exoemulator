"""restoring the trained model and predicting the cross section
"""

import matplotlib.pyplot as plt
from flax import nnx
import orbax.checkpoint as ocp
import jax
from jax import numpy as jnp
import numpy as np
from exoemulator.model.decoder import opaemulator_decoder
from exoemulator.train.opacity import OptExoJAX
from pathlib import Path


#ckpt_dir = Path("/home/kawahara/checkpoints")
ckpt_dir = Path("/home/kawahara/checkpoints_tmp")
#checkpointer = ocp.StandardCheckpointer()

# Restore the checkpoint back to its `nnx.State` structure - need an abstract reference.


opt = OptExoJAX()
graphdef, restored = opt.restore_state(opaemulator_decoder, ckpt_dir)
state_restored = restored.state
metadata = restored.metadata
nu_grid = restored.nu_grid
    

#state_restored = checkpointer.restore(ckpt_dir / 'state', abstract_state)
#nnx.display(state_restored)

emulator_model = nnx.merge(graphdef, state_restored)
input_par = jnp.array([729.0, jnp.log10(3.0e-1)])
output_vector = emulator_model(input_par)

xs_pred = opt.xs_prediction(output_vector)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(nu_grid, xs_pred, alpha=1, lw=2)
plt.yscale("log")
plt.savefig("restored.png")  # R: lerning rate 1e-4


