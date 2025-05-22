import os
import jax
import jax.numpy as jnp
import numpy as np
from jax import random
from flax import linen as nn
from flax.training import train_state
from flax.serialization import to_bytes
import optax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from codecarbon import EmissionsTracker
from tqdm import tqdm


class ResidualBlock(nn.Module):
    filters: int
    strides: int = 1

    @nn.compact
    def __call__(self, x, train=True):
        residual = x
        x = nn.Conv(self.filters, (3, 3), self.strides, padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        x = nn.Conv(self.filters, (3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)

        if residual.shape != x.shape:
            residual = nn.Conv(self.filters, (1, 1), self.strides, padding='SAME')(residual)
            residual = nn.BatchNorm(use_running_average=not train)(residual)

        return nn.relu(x + residual)


class ResNet18(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x, train=True):
        x = nn.Conv(64, (3, 3), padding='SAME')(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)

        for filters, strides in zip([64, 128, 256, 512], [1, 2, 2, 2]):
            x = ResidualBlock(filters, strides)(x, train)
            x = ResidualBlock(filters)(x, train)

        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes)(x)
        return x


def get_data_loaders(path, img_size=(32, 32), batch_size=128):
    datagen = ImageDataGenerator(rescale=1./255)
    train_gen = datagen.flow_from_directory(
        os.path.join(path, "train"), target_size=img_size,
        batch_size=batch_size, class_mode='categorical')
    test_gen = datagen.flow_from_directory(
        os.path.join(path, "test"), target_size=img_size,
        batch_size=batch_size, class_mode='categorical')
    return train_gen, test_gen, len(train_gen.class_indices)


def create_state(rng, model, learning_rate, input_shape):
    variables = model.init(rng, jnp.ones(input_shape), train=True)
    tx = optax.adam(learning_rate)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=variables['params'], tx=tx
    )
    return state, variables['batch_stats']


@jax.jit
def train_step(state, batch_stats, x, y):
    def loss_fn(params):
        outputs, new_model_state = state.apply_fn(
            {'params': params, 'batch_stats': batch_stats},
            x, train=True, mutable=['batch_stats']
        )
        loss = optax.softmax_cross_entropy(outputs, y).mean()
        return loss, (outputs, new_model_state['batch_stats'])

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, new_batch_stats)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    acc = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(y, -1))
    return state, new_batch_stats, loss, acc


@jax.jit
def eval_step(state, batch_stats, x, y):
    logits = state.apply_fn({'params': state.params, 'batch_stats': batch_stats}, x, train=False, mutable=False)
    loss = optax.softmax_cross_entropy(logits, y).mean()
    acc = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(y, -1))
    return loss, acc


def run_experiment(dataset_path, output_file, checkpoint_path, img_size=(32, 32), epochs=10, batch_size=128):
    rng = random.PRNGKey(0)
    train_gen, test_gen, num_classes = get_data_loaders(dataset_path, img_size, batch_size)
    model = ResNet18(num_classes=num_classes)
    state, batch_stats = create_state(rng, model, 1e-4, (1, *img_size, 3))
    tracker = EmissionsTracker(output_dir="python/jax/emissions", output_file=output_file, log_level="error")

    for epoch in range(epochs):
        tqdm.write(f"\nEpoch {epoch + 1}/{epochs}")
        tracker.start()

        train_losses, train_accs = [], []
        steps_per_epoch = len(train_gen)
        for _ in tqdm(range(steps_per_epoch), desc="Training", leave=True):
            x, y = next(train_gen)
            xb, yb = jnp.array(x), jnp.array(y)
            state, batch_stats, loss, acc = train_step(state, batch_stats, xb, yb)
            train_losses.append(loss)
            train_accs.append(acc)

        test_losses, test_accs = [], []
        for _ in tqdm(range(len(test_gen)), desc="Evaluating", leave=True):
            x, y = next(test_gen)
            if x.shape[0] != batch_size:
                continue
            xb, yb = jnp.array(x), jnp.array(y)
            loss, acc = eval_step(state, batch_stats, xb, yb)
            test_losses.append(loss)
            test_accs.append(acc)

        tracker.stop()

        tqdm.write(
            f"Train Loss={jnp.mean(jnp.array(train_losses)):.4f}, "
            f"Train Acc={jnp.mean(jnp.array(train_accs)) * 100:.2f}%, "
            f"Test Loss={jnp.mean(jnp.array(test_losses)):.4f}, "
            f"Test Acc={jnp.mean(jnp.array(test_accs)) * 100:.2f}%"
        )

        os.makedirs("checkpoints", exist_ok=True)
        with open(checkpoint_path, 'wb') as f:
            f.write(to_bytes(state.params))