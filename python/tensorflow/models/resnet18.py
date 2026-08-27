# ================== ALL-TF_KERAS + MODEL GARDEN ==================
import os
# Consigliato: esplicita backend Keras 3 prima degli import
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.deepgreen_bench import Harness, RunContext
from tools.deepgreen_loader import train_test_loaders
from tf_keras.preprocessing.image import ImageDataGenerator          # <-- tf_keras, non tensorflow.keras
from tf_keras import layers, models, optimizers, losses, metrics
import tensorflow as tf
import numpy as np
from official.vision.modeling.backbones import resnet as resnet_v1   # Model Garden

# ---------------- DATA ----------------
def get_loaders(dataset_path, img_size=(32, 32), batch_size=128, seed=None):
    """Delegate to the shared tf.data pipeline (spec S3).

    The first campaign used ImageDataGenerator.flow_from_directory, which decodes
    with PIL in the calling thread. Its effective concurrency was neither the 2
    workers used by PyTorch, C++ and Java nor knowable from the source, and the
    audit showed loader parallelism is the dominant confound in this workload
    (Spearman -0.73 against epoch duration, with the GPU at 24-56% of its power
    limit). tools/deepgreen_loader.py makes the thread count an explicit,
    identical number and applies the same preprocessing as every other stack.
    """
    train, test, num_classes = train_test_loaders(
        dataset_path, img_size=img_size, batch_size=batch_size, seed=seed, one_hot=True)
    return train, test, num_classes



from tf_keras import callbacks

class PercentProgbar(callbacks.ProgbarLogger):
    """Mostra accuracy in percentuale durante il fit."""
    def on_epoch_end(self, epoch, logs=None):
        # converti acc/val_acc in percentuale
        if logs is not None:
            if "acc" in logs:
                logs["acc"] = logs["acc"] * 100
            if "val_acc" in logs:
                logs["val_acc"] = logs["val_acc"] * 100
        super().on_epoch_end(epoch, logs)

    def on_train_batch_end(self, batch, logs=None):
        if logs is not None and "acc" in logs:
            logs["acc"] = logs["acc"] * 100
        super().on_train_batch_end(batch, logs)

    def on_test_batch_end(self, batch, logs=None):
        if logs is not None and "acc" in logs:
            logs["acc"] = logs["acc"] * 100
        super().on_test_batch_end(batch, logs)


# ---------------- MODEL: ResNet-18 (Model Garden) ----------------
def build_resnet18_garden(input_shape=(32, 32, 3), num_classes=100):
    """
    Backbone ResNet-18 dal Model Garden, testa con GAP + Dense softmax (one-hot).
    Tutto su tf_keras.
    """
    backbone = resnet_v1.ResNet(
        model_id=18,
        input_specs=layers.InputSpec(shape=(None, *input_shape)),  # Keras 3 InputSpec
        bn_trainable=True,
        se_ratio=None,           # ResNet "classico"
        stem_type='v0',          # stem standard
        resnetd_shortcut=False,
    )

    inputs = layers.Input(shape=input_shape)
    # training= is deliberately NOT pinned here. The submitted code hard-coded
    # training=True, which bakes training-mode behaviour into the functional
    # graph: batch normalisation then uses the statistics of the current batch
    # even during evaluation. The test split is served in class order, so each
    # evaluation batch is a single class and BN normalises against degenerate
    # per-class statistics. Every other ecosystem switches to eval mode
    # (model.eval(), net.set_train(false), $eval()), so this stack was both
    # scoring badly and measuring a different computation at inference time.
    feats = backbone(inputs)  # Keras propagates the correct mode per call

    # Se è dict, prendi l'ultimo stage
    if isinstance(feats, dict):
        last_key = sorted(feats.keys())[-1]
        x = feats[last_key]
    else:
        x = feats

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)  # one-hot ⇒ softmax
    return models.Model(inputs, outputs, name="resnet18_32_garden")

# ---------------- SANITY CHECKS ----------------
def sanity_checks(model, train_loader, test_loader):
    print("[CHECK] model.output_shape:", model.output_shape)
    train_classes = list(train_loader.class_indices.keys())
    test_classes  = list(test_loader.class_indices.keys())
    print(f"[CHECK] #classes train/test = {len(train_classes)} / {len(test_classes)}")
    if train_classes != test_classes:
        raise RuntimeError("Class order mismatch train vs test.")

    # Quick batch acc (deve essere > random)
    x_batch, y_batch = next(iter(train_loader.dataset))
    preds = model(x_batch, training=False)
    batch_acc = metrics.categorical_accuracy(y_batch, preds).numpy().mean()
    print(f"[CHECK] quick batch acc ≈ {batch_acc:.4f} (random ~ {1/len(train_classes):.4f})")

# ---------------- RUN ----------------
def run_experiment(dataset_path, output_file_train, output_file_eval, checkpoint_path,
                   img_size=(32, 32), epochs=30, batch_size=128, lr=1e-4, seed=None,
                   repetition=0, dataset_name=None, precision="fp32", arch="resnet18"):
    """Run one independent repetition of the resnet18 TensorFlow experiment.

    See tools/deepgreen_bench.py for why CodeCarbon is configured centrally and
    why accuracy is persisted. In the first campaign this stack used the
    CodeCarbon default 15 s sampling interval while the JAX stack used 1 s, and
    no quality metric was written to disk.
    """
    ctx = RunContext(
        ecosystem="Python/TensorFlow",
        model=arch,
        dataset=dataset_name or Path(dataset_path).name,
        repetition=repetition,
        seed=seed,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        precision=precision,
    )

    with Harness(ctx) as bench:
        bench.set_seeds()

        train_loader, test_loader, num_classes = get_loaders(dataset_path, img_size, batch_size, ctx.seed)
        model = build_resnet18_garden(input_shape=img_size + (3,), num_classes=num_classes)

        model.compile(
            optimizer=optimizers.Adam(learning_rate=lr),
            loss=losses.CategoricalCrossentropy(from_logits=False),
            metrics=[metrics.CategoricalAccuracy(name="acc")]
        )

        sanity_checks(model, train_loader, test_loader)

        steps_per_epoch = train_loader.samples // batch_size
        val_steps = test_loader.samples // batch_size

        os.makedirs(os.path.dirname(checkpoint_path) or "checkpoints", exist_ok=True)

        for epoch in range(1, epochs + 1):
            print(f"\n=== TRAIN epoch {epoch}/{epochs} (rep {ctx.repetition}, seed {ctx.seed}) ===")
            with bench.track("train", epoch):
                hist = model.fit(
                    train_loader.dataset,
                    epochs=1,
                    steps_per_epoch=steps_per_epoch,
                    callbacks=[PercentProgbar()],
                    verbose=2,
                )

            print(f"\n=== EVAL epoch {epoch}/{epochs} ===")
            with bench.track("eval", epoch):
                eval_out = model.evaluate(test_loader.dataset, steps=val_steps, verbose=0, return_dict=True)

            bench.log_metrics(
                epoch,
                train_loss=float(hist.history["loss"][-1]),
                train_acc=float(hist.history.get("acc", [float("nan")])[-1]),
                test_loss=float(eval_out.get("loss", float("nan"))),
                # percent, to match train_acc and every other ecosystem
                test_acc=100.0 * float(eval_out.get("acc", float("nan"))),
            )

        # Keras 3 rejects any suffix other than .weights.h5, while the other
        # stacks accept whatever path they are given. Normalise here so one
        # campaign driver can address every ecosystem the same way.
        if not str(checkpoint_path).endswith(".weights.h5"):
            checkpoint_path = os.path.splitext(str(checkpoint_path))[0] + ".weights.h5"
        model.save_weights(checkpoint_path)
