from flax import nnx
from functools import partial
from jax import jit
from jax import vmap
import jax.numpy as jnp
import orbax.checkpoint as ocp
import pathlib


class OpaEmulator:

    def __init__(self, model, ckpt_dir: pathlib.Path):
        """Initializes the OpaEmulator class

        Args:
            model: exoemulator model
            ckpt_dir (pathlib.Path): checkpoint directory
        """
        self.ckpt_dir = ckpt_dir
        self.emulator_model = self.restore_model(model)

    @partial(jit, static_argnums=(0,))
    def xsvector(self, T, P):
        """compute the cross section vector

        Args:
            T (float): temperature
            P (float): pressure

        Returns:
            array: cross section vector

        """
        input_par = jnp.array([T, jnp.log10(P)])
        output_vector = self.emulator_model(input_par)
        return 10 ** (output_vector / self.factor - self.offset)

    @partial(jit, static_argnums=(0,))
    def xsmatrix(self, Tarr, Parr):
        vmap_xsvector = vmap(self.xsvector,(0,0))
        return vmap_xsvector(Tarr, Parr)


    def restore_state(self, model):
        """restore the state of the model

        Args:
            model: exoemulator model

        Returns:
            graphdef, restored
        """
        checkpointer = ocp.Checkpointer(
            ocp.CompositeCheckpointHandler("state", "nu_grid", "metadata")
        )

        # load metadata only
        restored = checkpointer.restore(
            self.ckpt_dir / "state",
            args=ocp.args.Composite(
                metadata=ocp.args.JsonRestore(),
            ),
        )
        # generate an abstract model
        abstract_model = nnx.eval_shape(
            lambda: model(
                rngs=nnx.Rngs(0), grid_length=restored.metadata["grid_length"]
            )
        )
        graphdef, abstract_state = nnx.split(abstract_model)

        # restore the state
        restored = checkpointer.restore(
            self.ckpt_dir / "state",
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(abstract_state),
                nu_grid=ocp.args.ArrayRestore(),
                metadata=ocp.args.JsonRestore(),
            ),
        )

        self.trange = restored.metadata["trange"]
        self.prange = restored.metadata["prange"]
        self.offset = restored.metadata["offset"]
        self.factor = restored.metadata["factor"]

        return graphdef, restored

    def restore_model(self, model):
        """restores the emulator model

        Args:
            model: exoemulator model


        Returns:
            emulator_model
        """
        graphdef, restored = self.restore_state(model)
        self.nu_grid = restored.nu_grid

        emulator_model = nnx.merge(graphdef, restored.state)
        return emulator_model

    def xs_prediction(self, output_vector):
        """Predicts cross section from the output vector

        Args:
            output_vector (array): The output vector from the emulator model.

        Returns:
            array: The predicted cross section.
        """
