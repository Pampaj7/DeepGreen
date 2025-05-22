import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, optimizers
import os
from codecarbon import EmissionsTracker
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback


def build_vgg16(input_shape=(32, 32, 3), num_classes=100):
    base_model = VGG16(include_top=False, input_shape=input_shape, weights=None)
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dense(4096, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def get_loaders(dataset_path, img_size=(32, 32), batch_size=128):
    datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = datagen.flow_from_directory(
        os.path.join(dataset_path, "train"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    test_split = "test" if os.path.exists(os.path.join(dataset_path, "test")) else "val"
    test_generator = datagen.flow_from_directory(
        os.path.join(dataset_path, test_split),
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical'
    )

    return train_generator, test_generator, len(train_generator.class_indices)


def run_experiment(dataset_path, output_file, checkpoint_path, img_size=(32, 32), epochs=30, batch_size=128):
    train_loader, test_loader, num_classes = get_loaders(dataset_path, img_size, batch_size)
    model = build_vgg16(input_shape=img_size + (3,), num_classes=num_classes)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.Adam(learning_rate=1e-4),
        metrics=['accuracy']
    )

    model.fit(
        train_loader,
        validation_data=test_loader,
        epochs=epochs,
        callbacks=[
            CodeCarbonPerEpoch(
                base_output_dir="python/tensorflow/emissions/",
                base_output_file=output_file.replace(".csv", "")
            )
        ]
    )


    os.makedirs("checkpoints", exist_ok=True)
    model.save_weights(checkpoint_path)


class CodeCarbonPerEpoch(Callback):
    def __init__(self, base_output_dir="emissions", base_output_file="run"):
        super().__init__()
        self.base_output_dir = base_output_dir
        self.base_output_file = base_output_file
        self.tracker = None
        os.makedirs(self.base_output_dir, exist_ok=True)

    def on_epoch_begin(self, epoch, logs=None):
        self.tracker = EmissionsTracker(
            output_dir=self.base_output_dir,
            output_file=f"{self.base_output_file}_epoch_{epoch}.csv",
            log_level="error"
        )
        self.tracker.start()

    def on_epoch_end(self, epoch, logs=None):
        if self.tracker:
            self.tracker.stop()