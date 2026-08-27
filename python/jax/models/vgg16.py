import os
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn
from flax.training import train_state
from flax.serialization import to_bytes
import optax
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.deepgreen_bench import Harness, RunContext
from tools.deepgreen_loader import train_test_loaders
from tqdm import tqdm

# backbone community
from flaxmodels import VGG16 as FM_VGG16


# ===================== MODEL =====================
class VGG16_32(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, x, train: bool = True):
        # VGG16 backbone "stock" senza head
        backbone = FM_VGG16(
            pretrained=None,
            include_head=False,   # sblocca input arbitrari (32x32)
            output="activations",  # restituisce dict di feature
            normalize=False,      # usi già rescale 1/255 nel loader
        )
        feats = backbone(x, train=train)

        # Prendi l’ultimo blocco feature ("block5") o, se non presente, l'ultima mappa
        if isinstance(feats, dict):
            feats = feats.get("block5", list(feats.values())[-1])

        # Testa minima: GAP + Dense(num_classes)
        x = jnp.mean(feats, axis=(1, 2))      # Global Average Pooling
        x = nn.Dense(self.num_classes)(x)     # classificatore lineare
        return x

# ===================== DATA =====================
def get_data_loaders(dataset_path, img_size=(32, 32), batch_size=128, seed=None):
    """Delegate to the shared tf.data pipeline (spec S3).

    The first campaign used ImageDataGenerator.flow_from_directory and pulled
    batches with next(gen), decoding with PIL in the calling thread. Its
    effective concurrency was neither the 2 workers used by PyTorch, C++ and
    Java nor knowable from the source, and the audit showed loader parallelism
    is the dominant confound in this workload. tools/deepgreen_loader.py makes
    the thread count explicit and identical, with the same preprocessing.
    """
    train, test, num_classes = train_test_loaders(
        dataset_path, img_size=img_size, batch_size=batch_size, seed=seed, one_hot=True)
    return train, test, num_classes




# ===================== STATE / STEPS =====================
def create_state(rng, model, learning_rate, input_shape):
    """
    Inizializza parametri. (VGG16_32 non usa BatchNorm -> niente batch_stats)
    """
    variables = model.init(rng, jnp.ones(
        input_shape, dtype=jnp.float32), train=True)
    params = variables["params"]
    tx = optax.adam(learning_rate)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)
    return state


@jax.jit
def train_step(state, x, y):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, x,
                                train=True, mutable=False)
        loss = optax.softmax_cross_entropy(logits, y).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    acc = jnp.mean(jnp.argmax(logits, -1) == jnp.argmax(y, -1))
    return state, loss, acc


@jax.jit
def eval_step(state, x, y):
    logits = state.apply_fn({"params": state.params},
                            x, train=False, mutable=False)
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
    repetition=0,
    seed=None,
    dataset_name=None,
    precision="fp32",
):
    """Run one independent repetition of the vgg16 JAX experiment.

    In the first campaign this stack sampled CodeCarbon every 1 s while the
    PyTorch and TensorFlow stacks used the 15 s default, and no quality metric
    reached disk. Both are now handled centrally by tools/deepgreen_bench.py.
    """
    ctx = RunContext(
        ecosystem="Python/JAX",
        model="vgg16",
        dataset=dataset_name or Path(dataset_path).name,
        repetition=repetition,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        lr=learning_rate,
        precision=precision,
    )
    bench = Harness(ctx)
    bench.set_seeds()
    rng = random.PRNGKey(int(ctx.seed))
    train_gen, test_gen, num_classes = get_data_loaders(
        dataset_path, img_size, batch_size)

    # Modello community (flaxmodels) + testa custom per 32x32
    model = VGG16_32(num_classes=num_classes)
    state = create_state(rng, model, learning_rate, (1, *img_size, 3))

    os.makedirs("python/jax/emissions/", exist_ok=True)
    os.makedirs("checkpoints/", exist_ok=True)

    # Usa SOLO batch completi -> niente liste vuote
    steps_per_epoch = train_gen.samples // batch_size
    val_steps = test_gen.samples // batch_size

    def safe_mean(lst):
        if not lst:
            return jnp.nan
        return jnp.mean(jnp.stack(lst))

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        # tf.data iterators are single-pass; re-create them per epoch.
        train_iter = train_gen.as_numpy()
        test_iter = test_gen.as_numpy()

        # --- Emissioni fase TRAIN ---
        _train_cm = bench.track("train", epoch)
        _train_cm.__enter__()

        train_losses, train_accs = [], []
        for _ in tqdm(range(steps_per_epoch), desc=f"[Train] Epoch {epoch}"):
            x, y = next(train_iter)
            xb = jnp.asarray(x, dtype=jnp.float32)  # NHWC in [0,1]
            yb = jnp.asarray(y, dtype=jnp.float32)  # one-hot
            state, loss, acc = train_step(state, xb, yb)
            train_losses.append(loss)
            train_accs.append(acc)

        _train_cm.__exit__(None, None, None)

        # --- Emissioni fase EVAL ---
        _eval_cm = bench.track("eval", epoch)
        _eval_cm.__enter__()

        test_losses, test_accs = [], []
        for _ in tqdm(range(val_steps), desc=f"[Eval] Epoch {epoch}"):
            x, y = next(test_iter)
            xb = jnp.asarray(x, dtype=jnp.float32)
            yb = jnp.asarray(y, dtype=jnp.float32)
            loss, acc = eval_step(state, xb, yb)
            test_losses.append(loss)
            test_accs.append(acc)

        _eval_cm.__exit__(None, None, None)

        # --- Logging robusto ---
        tl = safe_mean(train_losses)
        ta = safe_mean(train_accs) * 100.0
        vl = safe_mean(test_losses)
        va = safe_mean(test_accs) * 100.0

        bench.log_metrics(epoch, train_loss=float(tl), train_acc=float(ta),
                          test_loss=float(vl), test_acc=float(va))
        print(f"Train Loss={tl:.4f}, Train Acc={ta:.2f}%, "
              f"Test Loss={vl:.4f}, Test Acc={va:.2f}%")

        # --- Salvataggio checkpoint (solo params: niente BN) ---
        with open(checkpoint_path, "wb") as f:
            f.write(to_bytes({"params": state.params}))

    bench.close()


# ===================== ESEMPIO USO =====================
# if __name__ == "__main__":
#     run_experiment(
#         dataset_path="/path/al/tuo/dataset",
#         output_file_base="jax_flaxmodels_vgg16_32",
#         checkpoint_path="checkpoints/jax_flaxmodels_vgg16_32.msgpack",
#         img_size=(32, 32),
#         epochs=30,
#         batch_size=128,
#         learning_rate=1e-4,
#     )
