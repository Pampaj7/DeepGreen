import jax
import jax.numpy as jnp
from jax import grad, jit
import numpy as np

# Dummy dataset: y = 3 * x + 1
key = jax.random.PRNGKey(0)
x_data = jax.random.normal(key, (1000,))
y_data = 3.0 * x_data + 1.0

# Model params
params = {'W': jnp.array(0.0), 'b': jnp.array(0.0)}

# Forward pass
def model(params, x):
    return params['W'] * x + params['b']

# Loss function: mean squared error
def loss_fn(params, x, y):
    preds = model(params, x)
    return jnp.mean((preds - y) ** 2)

# Optimizer (SGD)
@jit
def update(params, x, y, lr=1e-2):
    grads = grad(loss_fn)(params, x, y)
    return {
        'W': params['W'] - lr * grads['W'],
        'b': params['b'] - lr * grads['b']
    }

# Training loop
for epoch in range(10):
    params = update(params, x_data, y_data)
    loss = loss_fn(params, x_data, y_data)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}, W: {params['W']:.4f}, b: {params['b']:.4f}")