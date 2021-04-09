from tensorflow.keras.applications import Xception
from tensorflow.keras import models, layers, optimizers
import numpy as np
import matplotlib.pyplot as plt

from create_DS import X_train, y_train, X_val, y_val, X_test, y_test



Xception_arch = Xception(include_top = False, input_shape=(100,100,3))

model = models.Sequential()
model.add(Xception_arch)
model.add(layers.Flatten())
model.add(layers.Dense(256))
model.add(layers.Dense(23, activation="softmax"))
model.summary()
Xception_arch.trainable = False
