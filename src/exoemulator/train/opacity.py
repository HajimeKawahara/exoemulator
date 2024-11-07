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
import pathlib

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
        """ Predicts cross section from the output vector
        
        Args:
            output_vector (array): The output vector from the emulator model.

        Returns:
            array: The predicted cross section.
        """
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
        """ Trains the model for one step.
        Args:
            model (nnx.Module): The neural network model to be trained.
            optimizer (nnx.Optimizer): The optimizer used to update the model parameters.
            metric (nnx.MultiMetric): The metric object used to track training performance.
            input_parameter: The input data for the model (T,P).
            label_vector: The ground truth label vector (reference cross section) corresponding to the input data.

        Returns:
            float: The computed loss value for the current training step.
        """
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
        """ Evaluates the model on the test data.
        Args:
            model (nnx.Module): The neural network model to be evaluated.
            input_parameter (Any): The input data for the model.
            label_vector (Any): The true labels corresponding to the input data.
            
        Returns:
            float: The loss value computed for the given input and label.
        """
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
        """ Trains the model using the given parameters.
        
        Args:
            model (nnx.Module): The neural network model to be trained.
            metrics (nnx.MultiMetric): Metrics to evaluate the model's performance.
            trange (tuple): Range of temperature values for training data generation.
            prange (tuple): Range of pressure values for training data generation.
            learning_rate_arr (list): List of learning rates for each training phase.
            niter_arr (list): List of iteration counts for each training phase.
            nsample_minibatch (int, optional): Number of samples in each minibatch. Defaults to 100.
            adamw_momentum (float, optional): Momentum parameter for the AdamW optimizer. Defaults to 0.9.
            n_single_learn (int, optional): Number of iterations before generating a new batch of data. Defaults to 20.
            metric_save_interval (int, optional): Interval for saving metrics during training. Defaults to 100.
        """

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
            description = self._desc_for_tqdm(N_lr, j, learning_rate)
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

    def _desc_for_tqdm(self, N_lr, j, learning_rate):
        """ Generates a description for the tqdm progress bar."""
        description = (
            "learning rate: " + str(learning_rate) + ":" + str(j + 1) + "/" + str(N_lr)
        )

        return description

    def save_state(self, ckpt_dir: pathlib.Path, state):
        """save the state of the model
        
        Args:
            ckpt_dir: checkpoint directory
            state: state
        """
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

    

if __name__ == "__main__":
    opt = OptExoJAX()
