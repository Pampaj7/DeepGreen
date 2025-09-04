# ================== ALL-TF_KERAS + MODEL GARDEN ==================
import os
# Consigliato: esplicita backend Keras 3 prima degli import
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

from codecarbon import EmissionsTracker
from tf_keras.preprocessing.image import ImageDataGenerator          # <-- tf_keras, non tensorflow.keras
from tf_keras import layers, models, optimizers, losses, metrics
import tensorflow as tf
import numpy as np
from official.vision.modeling.backbones import resnet as resnet_v1   # Model Garden

# ---------------- DATA ----------------
def get_loaders(dataset_path, img_size=(32, 32), batch_size=128, seed=42):
    """
    ImageDataGenerator (one-hot) con classi allineate tra train e test.
    Tutto da tf_keras per evitare il crash in compile/fit.
    """
    datagen = ImageDataGenerator(rescale=1./255)

    train_dir = os.path.join(dataset_path, "train")
    test_split = "test" if os.path.exists(os.path.join(dataset_path, "test")) else "val"
    test_dir = os.path.join(dataset_path, test_split)

    # 1) Train: ricava mappa classi
    train_gen = datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",   # one-hot
        shuffle=True,
        seed=seed
    )

    # 2) Test: forza **stesso ordine classi** del train
    class_order = sorted(train_gen.class_indices.keys())
    test_gen = datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
        classes=class_order
    )

    # 3) Assert duri (se falliscono, il problema è nei folder)
    assert list(train_gen.class_indices.keys()) == class_order, "Train classes out-of-order"
    assert list(test_gen.class_indices.keys())  == class_order, "Test classes not aligned to train"

    return train_gen, test_gen, len(class_order)

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
    feats = backbone(inputs, training=True)  # può ritornare dict o Tensor a seconda della config

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
    x_batch, y_batch = next(train_loader)
    preds = model(x_batch, training=False)
    batch_acc = metrics.categorical_accuracy(y_batch, preds).numpy().mean()
    print(f"[CHECK] quick batch acc ≈ {batch_acc:.4f} (random ~ {1/len(train_classes):.4f})")

# ---------------- RUN ----------------
def run_experiment(dataset_path, output_file_train, output_file_eval, checkpoint_path,
                   img_size=(32, 32), epochs=30, batch_size=128, lr=1e-3, seed=42):

    # (Se TF-GPU non è configurato e vuoi evitare warning rumorosi)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    train_loader, test_loader, num_classes = get_loaders(dataset_path, img_size, batch_size, seed)
    model = build_resnet18_garden(input_shape=img_size + (3,), num_classes=num_classes)

    # Compile coerente con one-hot
    model.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss=losses.CategoricalCrossentropy(from_logits=False),
        metrics=[metrics.CategoricalAccuracy(name="acc")]
    )

    # Sanity checks (se qui va, non avrai più acc=0 "misteriosi")
    sanity_checks(model, train_loader, test_loader)

    steps_per_epoch = train_loader.samples // batch_size
    val_steps = test_loader.samples // batch_size

    os.makedirs("python/tensorflow/emissions/", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, epochs + 1):
        print(f"\n=== TRAIN epoch {epoch} ===")
        tr = EmissionsTracker(
            output_dir="python/tensorflow/emissions/",
            output_file=f"{output_file_train}_epoch{epoch}.csv",
            save_to_file=True,
            allow_multiple_runs=True
        )
        tr.start()
        model.fit(
            train_loader,
            epochs=1,                        # <-- SOLO 1 epoca
            steps_per_epoch=steps_per_epoch,
            callbacks=[PercentProgbar()]
        )
        tr.stop()

        print(f"\n=== EVAL epoch {epoch} ===")
        ev = EmissionsTracker(
            output_dir="python/tensorflow/emissions/",
            output_file=f"{output_file_eval}_epoch{epoch}.csv",
            save_to_file=True,
            allow_multiple_runs=True
        )
        ev.start()
        model.evaluate(test_loader, steps=val_steps)
        ev.stop()

