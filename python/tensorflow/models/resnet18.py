# ================== ALL-TF_KERAS + MODEL GARDEN ==================
import os
# Consigliato: esplicita backend Keras 3 prima degli import
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.deepgreen_bench import Harness, RunContext, assert_parameter_count
from tools.deepgreen_loader import train_test_loaders
from tools.torch_init import apply_torchvision_init
from tf_keras.preprocessing.image import ImageDataGenerator          # <-- tf_keras, non tensorflow.keras
from tf_keras import layers, models, optimizers, losses, metrics
import tensorflow as tf
import numpy as np

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


# ---------------- MODEL: ResNet-18, as torchvision defines it ----------------
def _basic_block(x, filters, stride, name):
    """torchvision's BasicBlock: 3x3-BN-ReLU, 3x3-BN, add, ReLU.

    The projection shortcut appears only where the shape changes -- stride != 1
    or a channel count that differs. Model Garden puts one on the first stage as
    well, where the channels already match at stride 1, and that single extra
    1x1 convolution with its BatchNorm is the +4,224 parameters and +128 running
    statistics that separated this stack from the other six.
    """
    shortcut = x
    if stride != 1 or x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, use_bias=False,
                                 name=f"{name}_down_conv")(x)
        shortcut = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f"{name}_down_bn")(shortcut)

    y = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False,
                      name=f"{name}_conv1")(x)
    y = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f"{name}_bn1")(y)
    y = layers.ReLU(name=f"{name}_relu1")(y)
    y = layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False,
                      name=f"{name}_conv2")(y)
    y = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f"{name}_bn2")(y)
    y = layers.Add(name=f"{name}_add")([y, shortcut])
    return layers.ReLU(name=f"{name}_relu2")(y)


def build_resnet18_garden(input_shape=(32, 32, 3), num_classes=100):
    """ResNet-18, layer for layer as torchvision defines it.

    This was Model Garden's resnet_v1.ResNet(model_id=18), which is a ResNet-18
    but not *the* ResNet-18 the other six stacks train: it carries an extra
    projection shortcut on the first stage, so this stack had 11,232,036
    parameters against everyone else's 11,227,812. Small -- 0.04% -- and still a
    structurally different residual block in a study whose entire claim is that
    the network is held constant while the ecosystem varies. The name is kept so
    the campaign driver and the checkpoint paths do not move.

    Convolutions carry no bias because a BatchNorm follows and would cancel it;
    the stem is 7x7 stride 2 with padding 3, then 3x3 max-pool at stride 2, as
    in torchvision -- not a CIFAR stem. models/MANIFEST.json holds the count
    this must match, and assert_parameter_count refuses the run if it does not.
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.ZeroPadding2D(3, name="stem_pad")(inputs)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False, name="stem_conv")(x)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)
    x = layers.ZeroPadding2D(1, name="stem_pool_pad")(x)
    x = layers.MaxPooling2D(3, strides=2, name="stem_pool")(x)

    for stage, (filters, stride) in enumerate(
            ((64, 1), (128, 2), (256, 2), (512, 2)), start=1):
        x = _basic_block(x, filters, stride, name=f"layer{stage}_0")
        x = _basic_block(x, filters, 1, name=f"layer{stage}_1")

    # training= is deliberately NOT pinned anywhere in this graph. The submitted
    # code hard-coded training=True, which bakes training-mode behaviour in:
    # batch normalisation then uses the statistics of the current batch even
    # during evaluation. The test split is served in class order, so each
    # evaluation batch is a single class and BN normalises against degenerate
    # per-class statistics. Every other ecosystem switches to eval mode
    # (model.eval(), net.set_train(false), $eval()), so this stack was both
    # scoring badly and measuring a different computation at inference time.
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="fc")(x)
    return models.Model(inputs, outputs, name="resnet18_32")

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
        # Keras defaults to glorot_uniform; the other six stacks use
        # torchvision's kaiming_normal_(fan_out) for convolutions. One epoch
        # of Fashion-MNIST showed the cost: 77.4% here against 86-88% for
        # every aligned stack.
        touched = apply_torchvision_init(model, seed=int(ctx.seed))
        if touched == 0:
            raise RuntimeError(
                'apply_torchvision_init matched no layers: the model would train\n'
                'from a different distribution than every other stack, silently.')
        # Trainable weights only: torch's model.parameters() excludes the
        # BatchNorm running statistics that Keras's count_params() includes.
        assert_parameter_count("resnet18", ctx.dataset,
                               sum(int(w.shape.num_elements())
                                   for w in model.trainable_weights))

        model.compile(
            optimizer=optimizers.Adam(learning_rate=lr),
            loss=losses.CategoricalCrossentropy(from_logits=False),
            metrics=[metrics.CategoricalAccuracy(name="acc")]
        )

        sanity_checks(model, train_loader, test_loader)

        # ceil, not floor: with drop_remainder=False the loader yields a final
        # partial batch, and floor would step past exactly the images the other
        # five stacks train and evaluate on.
        steps_per_epoch = -(-train_loader.samples // batch_size)
        val_steps = -(-test_loader.samples // batch_size)

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
