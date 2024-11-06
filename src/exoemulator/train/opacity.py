"""opacity model training (opt).

    - spectrum training (spt)

"""

from exoemulator.utils.checkpackage import check_installed
from exoemulator.train.sampling import latin_hypercube_sampling
from exoemulator.model.loss import loss_l2
from functools import partial
from flax import nnx
from flax.nnx import jit
import numpy as np
import jax.numpy as jnp
import tqdm
import optax
import orbax.checkpoint as ocp


class OptExoJAX:
    """Opacity Training with ExoJAX"""

    def __init__(self, opa=None):
        check_installed("exojax")
        self.set_opa(opa)
        self.offset = 22.0
        self.factor = 0.3

    def set_opa(self, opa):
        """sets opa (opacity class in ExoJAX)

        Args:
            opa : "opa" (opacity class) in ExoJAX

        Notes:
            opa is an opacity class in ExoJAX, OpaPremodit, OpaModit, or OpaDirect

        """
        if opa is None:
            print("opa (ExoJAX opacity class) has not been given yet")
        else:
            print("opa (ExoJAX opacity class) in opt: ", opa.__class__.__name__)
            self.opa = opa

    def generate_batch(self, trange, prange, nsample, method="lhs"):
        """generates batched samples of temperature, pressure, and cross section

        Args:
            trange (tuple or list): A tuple containing the minimum and maximum temperature range.
            prange (tuple or list): A tuple containing the minimum and maximum pressure range.
            nsample (int): The number of samples to generate.
            method (str, optional): The sampling method to use. Defaults to "lhs".

        Notes:
            "lhs" means Latin Hypercube Sampling.

        Raises:
            NotImplementedError: If the specified method is not implemented.

        Returns:
            tuple: A tuple containing arrays of (temperature, pressure), and cross section matrix.
        """
        if method == "lhs":
            # tarr is linear, parr is log
            samples = latin_hypercube_sampling(trange, np.log10(prange), nsample)
            return samples, self.factor * (
                jnp.log10(self.opa.xsmatrix(samples[:, 0], 10 ** (samples[:, 1])))
                + self.offset
            )

        else:
            raise NotImplementedError(f"Method {method} is not implemented")

    def xs_prediction(self, output_vector):
        return 10 ** (output_vector / self.factor - self.offset)

    @partial(jit, static_argnums=(0,))
    def train_step(
        self,
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        metric: nnx.MultiMetric,
        input_parameter,
        label_vector,
    ):
        grad_fn = nnx.value_and_grad(loss_l2, has_aux=True)
        (loss, _), grads = grad_fn(model, input_parameter, label_vector)
        metric.update(loss=loss)
        optimizer.update(grads)
        return loss

    @partial(jit, static_argnums=(0,))
    def evaluate_step(
        self,
        model: nnx.Module,
        input_parameter,
        label_vector,
    ):
        grad_fn = nnx.value_and_grad(loss_l2, has_aux=True)
        (loss, _), _ = grad_fn(model, input_parameter, label_vector)
        return loss

    def train(
        self,
        model: nnx.Module,
        metrics: nnx.MultiMetric,
        trange,
        prange,
        learning_rate_arr,
        niter_arr,
        nsample_minibatch=100,
        adamw_momentum=0.9,
        n_single_learn=20,
        metric_save_interval=100,
    ):

        self.lossarr = []
        self.testlossarr = []
        self.trange = trange
        self.prange = prange

        N_lr = len(learning_rate_arr)
        for j in range(N_lr):
            learning_rate = learning_rate_arr[
                j
            ]  # learning rate scheduling (step decay)
            optimizer = nnx.Optimizer(model, optax.adamw(learning_rate, adamw_momentum))
            description = self.desc_for_tqdm(N_lr, j, learning_rate)
            for i in tqdm.tqdm(range(niter_arr[j]), desc=description):
                if np.mod(i, n_single_learn) == 0:
                    input_parameters, logxs = self.generate_batch(
                        trange=self.trange,
                        prange=self.prange,
                        nsample=nsample_minibatch,
                        method="lhs",
                    )
                loss = self.train_step(
                    model, optimizer, metrics, input_parameters, logxs
                )
                if np.mod(i, metric_save_interval) == 0:
                    input_parameters, logxs = self.generate_batch(
                        trange=self.trange,
                        prange=self.prange,
                        nsample=nsample_minibatch,
                        method="lhs",
                    )
                    testloss = self.evaluate_step(model, input_parameters, logxs)
                    self.lossarr.append(loss)
                    self.testlossarr.append(testloss)

    def desc_for_tqdm(self, N_lr, j, learning_rate):
        description = (
            "learning rate: " + str(learning_rate) + ":" + str(j + 1) + "/" + str(N_lr)
        )

        return description

    def save_state(self, ckpt_dir, state):
        metadata = {
            "grid_length": len(self.opa.nu_grid),
            "offset": self.offset,
            "factor": self.factor,
            "trange": self.trange,
            "prange": self.prange,
        }
        checkpointer = ocp.Checkpointer(
            ocp.CompositeCheckpointHandler("state", "nu_grid", "metadata")
        )
        checkpointer.save(
            ckpt_dir / "state",
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(state),
                nu_grid=ocp.args.ArraySave(self.opa.nu_grid),
                metadata=ocp.args.JsonSave(metadata),
            ),
        )

    def restore_state(self, model, ckpt_dir):
        checkpointer = ocp.Checkpointer(
            ocp.CompositeCheckpointHandler("state", "nu_grid", "metadata")
        )

        # load metadata only
        restored = checkpointer.restore(
            ckpt_dir / "state",
            args=ocp.args.Composite(
                metadata=ocp.args.JsonRestore(),
            ),
        )
        grid_length = restored.metadata["grid_length"]
        abstract_model = nnx.eval_shape(
            lambda: model(rngs=nnx.Rngs(0), grid_length=grid_length)
        )
        graphdef, abstract_state = nnx.split(abstract_model)

        # restore the state
        restored = checkpointer.restore(
            ckpt_dir / "state",
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


if __name__ == "__main__":
    opt = OptExoJAX()
