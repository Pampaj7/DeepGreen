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

# --- Model ---
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
        return nn.Dense(self.num_classes)(x)

# --- Utilities ---
def get_data_loaders(path, img_size=(32, 32), batch_size=128):
    datagen = ImageDataGenerator(rescale=1. / 255)
    
    train_gen = datagen.flow_from_directory(
        os.path.join(path, "train"), target_size=img_size,
        batch_size=batch_size, class_mode='categorical'
    )
    
    # Fallback da 'test' a 'eval' se serve
    test_dir = os.path.join(path, "test")
    if not os.path.exists(test_dir):
        test_dir = os.path.join(path, "val")

    test_gen = datagen.flow_from_directory(
        test_dir, target_size=img_size,
        batch_size=batch_size, class_mode='categorical'
    )

    return train_gen, test_gen, len(train_gen.class_indices)


def create_state(rng, model, learning_rate, input_shape):
    variables = model.init(rng, jnp.ones(input_shape), train=True)
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx), variables['batch_stats']

@jax.jit
def train_step(state, batch_stats, x, y):
    def loss_fn(params):
        outputs, new_model_state = state.apply_fn(
            {'params': params, 'batch_stats': batch_stats}, x, train=True, mutable=['batch_stats']
        )
        loss = optax.softmax_cross_entropy(outputs, y).mean()
        return loss, (outputs, new_model_state['batch_stats'])
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, new_bs)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    acc = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(y, -1))
    return state, new_bs, loss, acc

@jax.jit
def eval_step(state, batch_stats, x, y):
    logits = state.apply_fn({'params': state.params, 'batch_stats': batch_stats}, x, train=False, mutable=False)
    loss = optax.softmax_cross_entropy(logits, y).mean()
    acc = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(y, -1))
    return loss, acc

def run_experiment(dataset_path, output_file_base, checkpoint_path, img_size=(32, 32), epochs=30, batch_size=128):
    rng = random.PRNGKey(0)
    train_gen, test_gen, num_classes = get_data_loaders(dataset_path, img_size, batch_size)
    model = ResNet18(num_classes=num_classes)
    state, batch_stats = create_state(rng, model, 1e-4, (1, *img_size, 3))

    os.makedirs("python/jax/emissions/", exist_ok=True)
    os.makedirs("checkpoints/", exist_ok=True)

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # --- Train phase emissions ---
        train_tracker = EmissionsTracker(
            output_dir="python/jax/emissions/",
            output_file=f"{output_file_base}_train_epoch{epoch}.csv",
            log_level="error",
            measure_power_secs=1,
            save_to_file=True,
            allow_multiple_runs=True
        )

        train_tracker.start()

        train_losses, train_accs = [], []
        for _ in tqdm(range(len(train_gen)), desc=f"[Train] Epoch {epoch}"):
            x, y = next(train_gen)
            xb, yb = jnp.array(x), jnp.array(y)
            state, batch_stats, loss, acc = train_step(state, batch_stats, xb, yb)
            train_losses.append(loss)
            train_accs.append(acc)

        train_tracker.stop()

        # --- Eval phase emissions ---
        eval_tracker = EmissionsTracker(
            output_dir="python/jax/emissions/",
            output_file=f"{output_file_base}_eval_epoch{epoch}.csv",
            log_level="error",
            measure_power_secs=1,
            save_to_file=True,
            allow_multiple_runs=True
        )

        eval_tracker.start()

        test_losses, test_accs = [], []
        for _ in tqdm(range(len(test_gen)), desc=f"[Eval] Epoch {epoch}"):
            x, y = next(test_gen)
            if x.shape[0] != batch_size:
                continue  # Skip incomplete batch
            xb, yb = jnp.array(x), jnp.array(y)
            loss, acc = eval_step(state, batch_stats, xb, yb)
            test_losses.append(loss)
            test_accs.append(acc)

        eval_tracker.stop()

        # --- Logging ---
        print(f"Train Loss={jnp.mean(jnp.array(train_losses)):.4f}, "
              f"Train Acc={jnp.mean(jnp.array(train_accs)) * 100:.2f}%, "
              f"Test Loss={jnp.mean(jnp.array(test_losses)):.4f}, "
              f"Test Acc={jnp.mean(jnp.array(test_accs)) * 100:.2f}%")
