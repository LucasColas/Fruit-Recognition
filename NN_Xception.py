from tensorflow.keras.applications import Xception
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




Xception_arch = Xception(include_top = False, input_shape=(100,100,3))

model = models.Sequential()
model.add(Xception_arch)
model.add(layers.Flatten())
model.add(layers.Dense(256))
model.add(layers.Dense(15, activation="softmax"))
model.summary()
print("trainable weights", len(model.trainable_weights))
Xception_arch.trainable = False
print("trainable weights", len(model.trainable_weights))

model.compile(optimizer=optimizers.Adam(), loss="categorical_crossentropy", metrics=["acc"])


model.fit(X, Y, batch_size=32, epochs=15, validation_data=(X_valid, Y_valid))
