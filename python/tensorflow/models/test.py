import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # commenta per provare su GPU
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

print(">>> Building ResNet50...")
model = ResNet50(include_top=False, input_tensor=Input(shape=(224, 224, 3)), weights=None)
model.summary()