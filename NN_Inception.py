from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.preprocessing import ImageDataGenerator
import numpy as np
import os
import matplotlib.pyplot as plt

from create_DS import X_train, y_train, X_val, y_val, main_path
folders_path_train = os.path.join(main_path, "Train")
folders_path_val = os.path.join(main_path, "Validation")
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip = True)
train_generator = train_datagen.flow_from_directory(folders_path_train, target_size=(100,100), batch_size=32, class_mode="categorical")
valid_generator = train_generator.flow_from_directory(folders_path_val, target_size=(100,100), batch_size=32, class_mode="categorical")

steps_per_epoch = train_generator.n//train_generator.batch_size
step_size_valid = valid_generator.n//valid_generator.batch_size
model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=32, validation_data=valid_generator, validation_steps=valid_generator)

"""

def one_hot(labels, dimension=15):
    results = np.zeros((len(labels), dimension))

    for i, label in enumerate(labels):
        results[i, label] = 1

    return results

X = np.array(X_train, dtype="float32").reshape(-1, 100,100,3)
#X //= 255
Y = one_hot(y_train)
print(Y)

X_valid = np.array(X_val, dtype='float32').reshape(-1, 100, 100,3)
#X_valid //= 255
Y_valid = one_hot(y_val)

X_enc = preprocess_input(X)
X_valid_enc = preprocess_input(X_valid)


Inception_arch = InceptionV3(include_top = False, input_shape=(100,100,3))
print("Inception_arch : ", Inception_arch)
model = models.Sequential()
model.add(Inception_arch)
print("model",model)
print("Inception_arch", Inception_arch)
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

"""
