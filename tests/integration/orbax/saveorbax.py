from flax import nnx
import orbax.checkpoint as ocp
import jax
from jax import numpy as jnp
import numpy as np

ckpt_dir = ocp.test_utils.erase_and_create_empty("/home/kawahara/tmp_checkpoints/")


class decoder(nnx.Module):
    def __init__(self, *, rngs: nnx.Rngs, grid_length: int):
        # def __init__(self, rngs: nnx.Rngs, grid_length: int):
        self.dense_entrance = nnx.Linear(in_features=2, out_features=16, rngs=rngs)
        self.dense_1 = nnx.Linear(in_features=16, out_features=32, rngs=rngs)
        self.dense_2 = nnx.Linear(in_features=32, out_features=256, rngs=rngs)
        self.dense_3 = nnx.Linear(in_features=256, out_features=512, rngs=rngs)
        self.dense_out = nnx.Linear(
            in_features=512, out_features=grid_length, rngs=rngs
        )

    def __call__(self, input_parameter):
        x = nnx.gelu(self.dense_entrance(input_parameter))
        x = nnx.gelu(self.dense_1(x))
        x = nnx.gelu(self.dense_2(x))
        x = nnx.gelu(self.dense_3(x))
        return self.dense_out(x)


class TwoLayerMLP(nnx.Module):
    def __init__(self, dim, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(dim, dim, rngs=rngs, use_bias=False)
        self.linear2 = nnx.Linear(dim, dim, rngs=rngs, use_bias=False)

    def __call__(self, x):
        x = self.linear1(x)
        return self.linear2(x)


# Instantiate the model and show we can run it.
# model = TwoLayerMLP(4, rngs=nnx.Rngs(0))
grid_length = 20000
model = decoder(rngs=nnx.Rngs(0), grid_length=grid_length)
# x = jax.random.normal(jax.random.key(42), (3, 4))
# assert model(x).shape == (3, 4)

_, state = nnx.split(model)
nnx.display(state)

# metadata = None
metadata = {"grid_length": grid_length}
nu_grid = jnp.ones(grid_length)
#nu_grid = np.ones(grid_length)

if metadata is not None:
    checkpointer = ocp.Checkpointer(ocp.CompositeCheckpointHandler("state", "nu_grid", "metadata"))
    checkpointer.save(
        ckpt_dir / "state",
        args=ocp.args.Composite(
            state=ocp.args.StandardSave(state), nu_grid=ocp.args.ArraySave(nu_grid), metadata=ocp.args.JsonSave(metadata)
        ),
    )
    
else:
    checkpointer = ocp.StandardCheckpointer()
    checkpointer.save(ckpt_dir / "state", state)
    checkpointer.wait_until_finished()