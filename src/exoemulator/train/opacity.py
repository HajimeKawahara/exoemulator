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

class OptExoJAX:
    """Opacity Training with ExoJAX"""

    def __init__(self, opa=None, emu=None):
        check_installed("exojax")
        self.set_opa(opa)
        
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
        offset = 22.0
        factor = 0.3
        if method == "lhs":
            # tarr is linear, parr is log
            samples = latin_hypercube_sampling(trange, np.log10(prange), nsample)
            return samples, factor*(jnp.log10(self.opa.xsmatrix(samples[:, 0], 10 ** (samples[:, 1]))) + offset)

        else:
            raise NotImplementedError(f"Method {method} is not implemented")

    

    # @nnx.jit
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
        (loss, output_vector), grads = grad_fn(model, input_parameter, label_vector)
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
        (loss, output_vector), grads = grad_fn(model, input_parameter, label_vector)
        return loss

if __name__ == "__main__":
    opt = OptExoJAX()
