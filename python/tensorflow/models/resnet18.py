import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import os
from codecarbon import EmissionsTracker
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# --- Custom ResNet18 Implementation --- following paper description
class ResnetBlock(tf.keras.Model):
    def __init__(self, channels, down_sample=False):
        super().__init__()
        strides = 2 if down_sample else 1

        self.conv1 = layers.Conv2D(channels, 3, strides=strides, padding="same", kernel_initializer="he_normal")
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(channels, 3, strides=1, padding="same", kernel_initializer="he_normal")
        self.bn2 = layers.BatchNormalization()

        self.down_sample = down_sample
        if down_sample:
            self.res_conv = layers.Conv2D(channels, 1, strides=2, padding="same", kernel_initializer="he_normal")
            self.res_bn = layers.BatchNormalization()

    def call(self, inputs, training=False):
        residual = inputs
        x = tf.nn.relu(self.bn1(self.conv1(inputs), training=training))
        x = self.bn2(self.conv2(x), training=training)

        if self.down_sample:
            residual = self.res_bn(self.res_conv(residual), training=training)

        x += residual
        return tf.nn.relu(x)


class ResNet18(tf.keras.Model):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = layers.Conv2D(64, 7, strides=2, padding="same", kernel_initializer="he_normal")
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D(3, strides=2, padding="same")

        self.layer1 = [ResnetBlock(64), ResnetBlock(64)]
        self.layer2 = [ResnetBlock(128, down_sample=True), ResnetBlock(128)]
        self.layer3 = [ResnetBlock(256, down_sample=True), ResnetBlock(256)]
        self.layer4 = [ResnetBlock(512, down_sample=True), ResnetBlock(512)]

        self.gap = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(num_classes, activation="softmax")

    def call(self, x, training=False):
        x = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        x = self.pool1(x)
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
            for block in layer:
                x = block(x, training=training)
        x = self.gap(x)
        return self.fc(x)


def build_resnet18(input_shape=(32, 32, 3), num_classes=100):
    model = ResNet18(num_classes)
    model(tf.random.uniform((1, *input_shape)))  # forza la costruzione completa
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
    model = build_resnet18(input_shape=img_size + (3,), num_classes=num_classes)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.Adam(learning_rate=1e-4),
        metrics=['accuracy']
    )

    tracker = EmissionsTracker(output_dir="python/tensorflow/emissions/", output_file=output_file)
    tracker.start()

    model.fit(
        train_loader,
        validation_data=test_loader,
        epochs=epochs
    )

    tracker.stop()

    os.makedirs("checkpoints", exist_ok=True)
    model.save_weights(checkpoint_path)
