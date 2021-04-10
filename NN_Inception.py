from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras import models, layers, optimizers
import numpy as np
import matplotlib.pyplot as plt

from create_DS import X_train, y_train, X_val, y_val

X = np.array(X_train, dtype="float32").reshape(-1, 100,100,3)
X //= 255
Y = np.array(y_train)

X_valid = np.array(X_val, dtype='float32').reshape(-1, 100, 100,3)
X_valid //= 255
Y_valid = np.array(y_val)

X_enc = preprocess_input(X)
X_valid_enc = preprocess_input(X_valid)


Inception_arch = InceptionV3(include_top = False, input_shape=(100,100,3))

model = models.Sequential()
model.add(Inception_arch)
model.add(layers.Flatten())
model.add(layers.Dense(256))
model.add(layers.Dense(15, activation="softmax"))
model.summary()
print("trainable weights", len(model.trainable_weights))
Inception_arch.trainable = False
print("trainable weights", len(model.trainable_weights))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5), loss="categorical_crossentropy", metrics=["acc"])

print("X_enc : ",X_enc.shape, "X_val end : ",X_valid_enc.shape)
print("Y : ", Y.shape, "Y val : ", Y_valid.shape)

model.fit(X_enc, Y, batch_size=32, epochs=15, validation_data=(X_valid_enc, Y_valid))
