import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # usa CPU se vuoi disabilitare la GPU
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
import numpy as np

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

print(">>> Building ResNet50...")
base_model = ResNet50(include_top=False, input_tensor=Input(shape=(224, 224, 3)), weights=None)
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(10, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=x)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print(">>> Generating dummy data...")
X = np.random.rand(64, 224, 224, 3).astype(np.float32)
y = np.random.randint(0, 10, 64)

print(">>> Starting training...")
model.fit(X, y, epochs=3, batch_size=16)