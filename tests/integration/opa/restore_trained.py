"""restoring the trained model and predicting the cross section
"""

import matplotlib.pyplot as plt
from flax import nnx
import orbax.checkpoint as ocp
import jax
from jax import numpy as jnp
import numpy as np
from exoemulator.model.decoder import EmuMlpDecoder
from pathlib import Path

ckpt_dir = Path("/home/kawahara/checkpoints")
checkpointer = ocp.StandardCheckpointer()

# Restore the checkpoint back to its `nnx.State` structure - need an abstract reference.
abstract_model = nnx.eval_shape(lambda: EmuMlpDecoder(rngs=nnx.Rngs(0), grid_length=20000))
graphdef, abstract_state = nnx.split(abstract_model)
nnx.display(abstract_state)

state_restored = checkpointer.restore(ckpt_dir / 'state', abstract_state)
#nnx.display(state_restored)

emulator_model = nnx.merge(graphdef, state_restored)
input_par = jnp.array([729.0, jnp.log10(3.0e-1)])
output_vector = emulator_model(input_par)
offset = 22.0
factor = 0.3
def xs_prediction(output_vector, offset, factor):
    return 10 ** (output_vector / factor - offset)

xs_pred = xs_prediction(output_vector, offset, factor)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(xs_pred, alpha=1, lw=2)
plt.yscale("log")
plt.savefig("restored.png")  # R: lerning rate 1e-4


