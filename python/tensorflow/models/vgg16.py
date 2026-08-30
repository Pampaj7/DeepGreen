import tensorflow as tf
# tf_keras (Keras 2.19), the same implementation python/tensorflow/models/
# resnet18.py uses. This file used tensorflow.keras -- Keras 3.15 -- for the
# model and tf_keras for its callbacks, so one ecosystem was two: two Keras
# major versions across the study's two architectures, and inside this file a
# Keras 2 ProgbarLogger attached to a Keras 3 fit, which works only because
# Keras 3's ProgbarLogger does not accept the count_mode it is given.
from tf_keras.applications import VGG16
from tf_keras import layers, models, optimizers, losses, metrics
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tools.deepgreen_bench import Harness, RunContext, assert_parameter_count
from tools.deepgreen_loader import train_test_loaders
from tools.torch_init import apply_torchvision_init
from tf_keras.preprocessing.image import ImageDataGenerator
from tf_keras.callbacks import Callback


def build_vgg16(input_shape=(32, 32, 3), num_classes=100):
    """VGG-16 with the classifier the other six stacks use.

    This was Flatten + 4096 + 4096 + softmax on a 1x1x512 map -- 34,006,948
    parameters, against 14,765,988 in JAX and 134,670,244 in the LibTorch
    lineage. Four heads, one name, and the energy difference between them was
    being reported as a difference between ecosystems. The definition lives in
    scripts/export_torchscript_models.py and the count in models/MANIFEST.json,
    which build_and_check() asserts against at startup.

    Dropout is included because the reference head has it; it was absent here
    and present in every torch stack, which is a difference in what is computed
    and not only in how much.
    """
    base_model = VGG16(include_top=False, input_shape=input_shape, weights=None)
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax'),
    ])
    return model


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
    def __init__(self):
        super().__init__(count_mode="steps")  # <-- serve per abilitare la barra

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            if "accuracy" in logs:
                logs["accuracy"] *= 100
            if "val_accuracy" in logs:
                logs["val_accuracy"] *= 100
        super().on_epoch_end(epoch, logs)

    def on_train_batch_end(self, batch, logs=None):
        if logs and "accuracy" in logs:
            logs["accuracy"] *= 100
        super().on_train_batch_end(batch, logs)

    def on_test_batch_end(self, batch, logs=None):
        if logs and "accuracy" in logs:
            logs["accuracy"] *= 100
        super().on_test_batch_end(batch, logs)



def sanity_checks(model, train_loader, test_loader):
    """Fail fast on a class-order mismatch between the train and test splits.

    Mirrors the check in the ResNet-18 module. A silent mismatch produces a
    model that trains normally but evaluates at chance, which would make the
    energy-per-accuracy figures meaningless.
    """
    print("[CHECK] model.output_shape:", model.output_shape)
    train_classes = list(train_loader.class_indices.keys())
    test_classes = list(test_loader.class_indices.keys())
    print(f"[CHECK] #classes train/test = {len(train_classes)} / {len(test_classes)}")
    if train_classes != test_classes:
        raise RuntimeError("Class order mismatch train vs test.")


# ---------------- RUN ----------------
def run_experiment(dataset_path, output_file_train, output_file_eval, checkpoint_path,
                   img_size=(32, 32), epochs=30, batch_size=128, lr=1e-4, seed=None,
                   repetition=0, dataset_name=None, precision="fp32", arch="vgg16"):
    """Run one independent repetition of the vgg16 TensorFlow experiment.

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
        model = build_vgg16(input_shape=img_size + (3,), num_classes=num_classes)
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
        assert_parameter_count("vgg16", ctx.dataset,
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
