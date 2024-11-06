import jax.numpy as jnp
    
def loss_l2(model, input_parameter, label_vector):
    output_vector = model(input_parameter)
    loss = jnp.mean((output_vector - label_vector) ** 2)
    return loss, output_vector
