from flax import nnx

import jax.numpy as jnp

rngs = nnx.Rngs(0)

x = jnp.ones((1, 20, 1))
x = x.at[0, 10, 0].set(2.0)

# valid padding
exansion = 3
layer = nnx.ConvTranspose(
    in_features=1,
    out_features=1,
    strides=exansion,
    kernel_size=(1,),
    padding="SAME",
    rngs=rngs,
)

#print(layer.kernel.value.shape)
#print(layer.bias.value.shape)
out = layer(x)
print(x.shape, "->", out.shape)

# connection test
dense = nnx.Linear(in_features=2, out_features=20, rngs=rngs)
# y = jnp.array([1.0,2.0])
y = jnp.ones((1, 2))
intermediate = dense(y)
out = layer(intermediate[:, :, jnp.newaxis])
print(y.shape, "->", intermediate.shape, "->", out.shape)

#
import matplotlib.pyplot as plt
plt.plot(out[0, :, 0])
# plt.plot(out[0, :, 1])
# plt.plot(out[0, :, 2])
plt.savefig("convtranspose.png")
