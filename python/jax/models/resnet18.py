import os
import jax
import jax.numpy as jnp
from jax import random
from flax.training import train_state
from flax.serialization import to_bytes
import optax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from codecarbon import EmissionsTracker
from tqdm import tqdm

from flaxmodels import ResNet18 as FMResNet18


# ===================== DATA =====================
def get_data_loaders(path, img_size=(32, 32), batch_size=128):
    datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_gen = datagen.flow_from_directory(
        os.path.join(path, "train"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
    )

    test_dir = os.path.join(path, "test")
    if not os.path.exists(test_dir):
        test_dir = os.path.join(path, "val")

    test_gen = datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    num_classes = len(train_gen.class_indices)
    return train_gen, test_gen, num_classes


# ===================== STATE / STEPS =====================
def create_state(rng, model, learning_rate, input_shape):
    # ResNet18 usa BatchNorm -> 'batch_stats' presente
    variables = model.init(rng, jnp.ones(
        input_shape, dtype=jnp.float32), train=True)
    params = variables["params"]
    batch_stats = variables.get("batch_stats")

    tx = optax.adam(learning_rate)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)
    return state, batch_stats


@jax.jit
def train_step(state, batch_stats, x, y):
    def loss_fn(params):
        logits, new_state = state.apply_fn(
            {"params": params, "batch_stats": batch_stats},
            x, train=True, mutable=["batch_stats"]
        )
        loss = optax.softmax_cross_entropy(logits, y).mean()
        return loss, (logits, new_state["batch_stats"])

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, new_bs)), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    acc = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(y, -1))
    return state, new_bs, loss, acc


@jax.jit
def eval_step(state, batch_stats, x, y):
    logits = state.apply_fn(
        {"params": state.params, "batch_stats": batch_stats},
        x, train=False, mutable=False
    )
    loss = optax.softmax_cross_entropy(logits, y).mean()
    acc = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(y, -1))
    return loss, acc


# ===================== RUN =====================
def run_experiment(
    dataset_path,
    output_file_base,
    checkpoint_path,
    img_size=(32, 32),
    epochs=30,
    batch_size=128,
    learning_rate=1e-4,
):
    rng = random.PRNGKey(0)
    train_gen, test_gen, num_classes = get_data_loaders(
        dataset_path, img_size, batch_size)

    # ======= MODELLO PIÙ NATIVO POSSIBILE =======
    # ResNet18 community, head stock (GAP+Dense), nessun wrapper
    model = FMResNet18(
        output="logits",
        num_classes=num_classes,
        pretrained=None,
        normalize=False,     # lasciamo il rescale 1/255 del loader
    )
    state, batch_stats = create_state(
        rng, model, learning_rate, (1, *img_size, 3))

    os.makedirs("python/jax/emissions/", exist_ok=True)
    os.makedirs("checkpoints/", exist_ok=True)

    # Usa solo batch completi (niente glitch di medie su liste vuote)
    steps_per_epoch = train_gen.samples // batch_size
    val_steps = test_gen.samples // batch_size

    def safe_mean(lst):
        if not lst:
            return jnp.nan
        return jnp.mean(jnp.stack(lst))

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")

        # --- Emissioni TRAIN ---
        train_tracker = EmissionsTracker(
            output_dir="python/jax/emissions/",
            output_file=f"{output_file_base}_train_epoch{epoch}.csv",
            log_level="error",
            measure_power_secs=1,
            save_to_file=True,
            allow_multiple_runs=True,
        )
        train_tracker.start()

        train_losses, train_accs = [], []
        for _ in tqdm(range(steps_per_epoch), desc=f"[Train] Epoch {epoch}"):
            x, y = next(train_gen)
            xb = jnp.asarray(x, dtype=jnp.float32)
            yb = jnp.asarray(y, dtype=jnp.float32)
            state, batch_stats, loss, acc = train_step(
                state, batch_stats, xb, yb)
            train_losses.append(loss)
            train_accs.append(acc)

        train_tracker.stop()

        # --- Emissioni EVAL ---
        eval_tracker = EmissionsTracker(
            output_dir="python/jax/emissions/",
            output_file=f"{output_file_base}_eval_epoch{epoch}.csv",
            log_level="error",
            measure_power_secs=1,
            save_to_file=True,
            allow_multiple_runs=True,
        )
        eval_tracker.start()

        test_losses, test_accs = [], []
        for _ in tqdm(range(val_steps), desc=f"[Eval] Epoch {epoch}"):
            x, y = next(test_gen)
            xb = jnp.asarray(x, dtype=jnp.float32)
            yb = jnp.asarray(y, dtype=jnp.float32)
            loss, acc = eval_step(state, batch_stats, xb, yb)
            test_losses.append(loss)
            test_accs.append(acc)

        eval_tracker.stop()

        # --- Logging robusto ---
        tl = safe_mean(train_losses)
        ta = safe_mean(train_accs) * 100.0
        vl = safe_mean(test_losses)
        va = safe_mean(test_accs) * 100.0

        print(f"Train Loss={tl:.4f}, Train Acc={ta:.2f}%, "
              f"Test Loss={vl:.4f}, Test Acc={va:.2f}%")

        # --- Checkpoint (params + BN) ---
        with open(checkpoint_path, "wb") as f:
            f.write(
                to_bytes({"params": state.params, "batch_stats": batch_stats}))
