import tensorflow_models as tfm  # <- Model Garden
from codecarbon import EmissionsTracker
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tf_keras import layers, models, optimizers
import tensorflow as tf
import os
from official.vision.modeling.backbones import resnet as resnet_v1

# ---------------- DATA ----------------

def get_loaders(dataset_path, img_size=(32, 32), batch_size=128):
    datagen = ImageDataGenerator(rescale=1./255)
    train_gen = datagen.flow_from_directory(
        os.path.join(dataset_path, "train"),
        target_size=img_size, batch_size=batch_size, class_mode="categorical", shuffle=True
    )
    test_split = "test" if os.path.exists(
        os.path.join(dataset_path, "test")) else "val"
    test_gen = datagen.flow_from_directory(
        os.path.join(dataset_path, test_split),
        target_size=img_size, batch_size=batch_size, class_mode="categorical", shuffle=False
    )
    return train_gen, test_gen, len(train_gen.class_indices)


# ---------------- MODEL: ResNet18 (Model Garden) ----------------

def build_resnet18_garden(input_shape=(32, 32, 3), num_classes=100):
    # Backbone ResNet-18 del Model Garden (input_specs DEVE essere un InputSpec tf_keras)
    backbone = resnet_v1.ResNet(
        model_id=18,
        # oppure (None, None, None, 3)
        input_specs=layers.InputSpec(shape=(None, *input_shape)),
        bn_trainable=True,
        se_ratio=None,           # ResNet "classico"
        stem_type='v0',          # stem standard
        resnetd_shortcut=False,
    )

    # Functional API, sempre da tf_keras
    inputs = layers.Input(shape=input_shape)
    feats = backbone(inputs, training=True)  # Garden ritorna DICT o Tensor

    if isinstance(feats, dict):
        # prendi l’ultimo stage
        last_key = sorted(feats.keys())[-1]
        x = feats[last_key]
    else:
        x = feats

    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name="resnet18_32_garden")
# ---------------- RUN ----------------


def run_experiment(dataset_path, output_file_train, output_file_eval, checkpoint_path,
                   img_size=(32, 32), epochs=30, batch_size=128, lr=1e-4):
    train_loader, test_loader, num_classes = get_loaders(
        dataset_path, img_size, batch_size)
    model = build_resnet18_garden(
        input_shape=img_size + (3,), num_classes=num_classes)

    model.compile(
        optimizer=optimizers.Adam(lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    steps_per_epoch = train_loader.samples // batch_size
    val_steps = test_loader.samples // batch_size

    os.makedirs("python/tensorflow/emissions/", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(1, epochs + 1):
        # --- TRAIN emissions ---
        tr = EmissionsTracker(output_dir="python/tensorflow/emissions/",
                              output_file=f"{output_file_train}_epoch{epoch}.csv",
                              measure_power_secs=1, save_to_file=True, allow_multiple_runs=True)
        tr.start()
        model.fit(train_loader, epochs=1, steps_per_epoch=steps_per_epoch,
                  validation_data=None, verbose=1, initial_epoch=epoch-1)
        tr.stop()

        # --- EVAL emissions ---
        ev = EmissionsTracker(output_dir="python/tensorflow/emissions/",
                              output_file=f"{output_file_eval}_epoch{epoch}.csv",
                              measure_power_secs=1, save_to_file=True, allow_multiple_runs=True)
        ev.start()
        model.evaluate(test_loader, steps=val_steps, verbose=1)
        ev.stop()

    model.save_weights(checkpoint_path)
