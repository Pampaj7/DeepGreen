import keras
from keras.applications import VGG16
from keras import layers, models
from keras.optimizers import adam_v2
import os, sys


def export_vgg16(input_shape=(32, 32, 3), output_name = "sequential_vgg16_cifar100.h5", num_classes: int = 100, pretrained_weights = None):
	# base model
	base_model = VGG16(include_top=False, input_shape=input_shape, weights=pretrained_weights)

	# Convert to sequential model
	model = models.Sequential()
	for layer in base_model.layers:
		model.add(layer)

	# add fc layers
	model.add(layers.Flatten())
	model.add(layers.Dense(4096, activation='relu'))
	model.add(layers.Dense(4096, activation='relu'))
	model.add(layers.Dense(num_classes, activation='softmax'))

	model.compile(
		loss='categorical_crossentropy', # converted to LossLayer with MCXENT loss function
		optimizer=adam_v2.Adam(learning_rate=1e-5),
		metrics=['accuracy']
	)

	model.summary()

	model.save(output_name) # extention .keras is not usable by DL4J
	print(output_name + " saved at: " + os.getcwd())


if __name__ == "__main__":
	#print(keras.__version__)
	params = {}
	if len(sys.argv) > 1:
		params["output_name"] = sys.argv[1]
	if len(sys.argv) > 2:
		params["num_classes"] = int(sys.argv[2])
	#params["pretrained_weights"] = models.VGG16_Weights.IMAGENET1K_V1

	export_vgg16(**(params))