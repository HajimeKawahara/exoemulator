from curses import meta
from flax import nnx
import orbax.checkpoint as ocp
import jax
from jax import numpy as jnp
import numpy as np
from exoemulator.model.decoder import EmuMlpDecoder


class TwoLayerMLP(nnx.Module):
    def __init__(self, dim, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(dim, dim, rngs=rngs, use_bias=False)
        self.linear2 = nnx.Linear(dim, dim, rngs=rngs, use_bias=False)

    def __call__(self, x):
        x = self.linear1(x)
        return self.linear2(x)


from pathlib import Path

ckpt_dir = Path("/home/kawahara/tmp_checkpoints/")
# ckpt_dir = Path("/home/kawahara/checkpoints")


# Restore the checkpoint back to its `nnx.State` structure - need an abstract reference.
# abstract_model = nnx.eval_shape(lambda: TwoLayerMLP(4, rngs=nnx.Rngs(0)))
abstract_model = nnx.eval_shape(
    lambda: EmuMlpDecoder(rngs=nnx.Rngs(0), grid_length=20000)
)
graphdef, abstract_state = nnx.split(abstract_model)
# print('The abstract NNX state (all leaves are abstract arrays):')
# nnx.display(abstract_state)

metadata_store = True

if metadata_store:
    checkpointer = ocp.Checkpointer(ocp.CompositeCheckpointHandler("state", "metadata"))
    restored = checkpointer.restore(
        ckpt_dir / "state",
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(abstract_state),
            metadata=ocp.args.JsonRestore(),
        ),
    )
    state_restored = restored.state
    metadata = restored.metadata
    print(metadata)

else:
    checkpointer = ocp.StandardCheckpointer()
    state_restored = checkpointer.restore(ckpt_dir / "state", abstract_state)

#print(state_restored)

# jax.tree.map(np.testing.assert_array_equal, state, state_restored)
# print('NNX State restored: ')
# nnx.display(state_restored)

# The model is now good to use!
# model = nnx.merge(graphdef, state_restored)
# assert model(x).shape == (3, 4)
