import keras
from keras import layers, Model
import os, sys


def resnet_block(inputs, filters, downsample=False, name_prefix=""):
	shortcut = inputs
	strides = 2 if downsample else 1

	# Main path
	x = layers.Conv2D(filters, 3, strides=strides, padding="same", kernel_initializer="he_normal", name=f"{name_prefix}_conv1")(inputs)
	x = layers.BatchNormalization(name=f"{name_prefix}_bn1")(x)
	x = layers.ReLU(name=f"{name_prefix}_relu1")(x)

	x = layers.Conv2D(filters, 3, strides=1, padding="same", kernel_initializer="he_normal", name=f"{name_prefix}_conv2")(x)
	x = layers.BatchNormalization(name=f"{name_prefix}_bn2")(x)

	# Shortcut path
	if downsample or shortcut.shape[-1] != filters:
		shortcut = layers.Conv2D(filters, 1, strides=strides, padding="same", kernel_initializer="he_normal", name=f"{name_prefix}_shortcut_conv")(shortcut)
		shortcut = layers.BatchNormalization(name=f"{name_prefix}_shortcut_bn")(shortcut)

	# Merge
	x = layers.Add(name=f"{name_prefix}_add")([x, shortcut])
	x = layers.ReLU(name=f"{name_prefix}_relu_out")(x)
	return x


def build_resnet18(input_shape, num_classes):
	inputs = keras.Input(shape=input_shape, name="input")

	# Initial conv layer
	x = layers.Conv2D(64, 7, strides=2, padding="same", kernel_initializer="he_normal", name="conv1")(inputs)
	x = layers.BatchNormalization(name="bn1")(x)
	x = layers.ReLU(name="relu1")(x)
	x = layers.MaxPooling2D(pool_size=3, strides=2, padding="same", name="maxpool")(x)

	# 4 blocks: 2 for each group
	x = resnet_block(x, 64, name_prefix="block1_1")
	x = resnet_block(x, 64, name_prefix="block1_2")

	x = resnet_block(x, 128, downsample=True, name_prefix="block2_1")
	x = resnet_block(x, 128, name_prefix="block2_2")

	x = resnet_block(x, 256, downsample=True, name_prefix="block3_1")
	x = resnet_block(x, 256, name_prefix="block3_2")

	x = resnet_block(x, 512, downsample=True, name_prefix="block4_1")
	x = resnet_block(x, 512, name_prefix="block4_2")

	# Output layer
	x = layers.GlobalAveragePooling2D(name="avgpool")(x)
	outputs = layers.Dense(num_classes, activation="softmax", name="fc")(x)

	model = Model(inputs, outputs, name="ResNet18")
	return model


def export_resnet18(input_shape=(32, 32, 3), output_name = "resnet18_cifar100.h5", num_classes: int = 100):

	model = build_resnet18(input_shape, num_classes)

	model.compile(
		loss='categorical_crossentropy', # converted to LossLayer with MCXENT loss function
		optimizer=keras.optimizers.Adam(learning_rate=1e-3),
		metrics=['accuracy']
	)

	model.summary()

	model.save(output_name) # .keras is not usable by DL4J
	print(output_name + " saved at: " + os.getcwd())


if __name__ == "__main__":
	#print(keras.__version__)
	params = {}
	if len(sys.argv) > 1:
		params["output_name"] = sys.argv[1]
	if len(sys.argv) > 2:
		params["num_classes"] = int(sys.argv[2])
	#params["pretrained_weights"] = models.VGG16_Weights.IMAGENET1K_V1
	
	export_resnet18(**(params))
