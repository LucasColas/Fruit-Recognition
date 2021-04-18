from tensorflow.keras.applications import InceptionV3, VGG16
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import os
import matplotlib.pyplot as plt

#from create_DS import X_train, y_train, X_val, y_val, main_path
main_path = r'E:\Projets code\Dataset Fruit recognition'
folders_path_train = os.path.join(main_path, "Train")
folders_path_val = os.path.join(main_path, "Validation")
print("folders valid",folders_path_val)
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=42, brightness_range=(0.3,2), shear_range=0.2, zoom_range=0.2, horizontal_flip = True, vertical_flip = True)
train_generator = train_datagen.flow_from_directory(folders_path_train, target_size=(100,100), batch_size=32, class_mode="categorical")
valid_generator = train_datagen.flow_from_directory(folders_path_val, target_size=(100,100), batch_size=32, class_mode="categorical")

"""
for batch, label in train_generator:
    count = 0
    for index, image in enumerate(batch):
        plt.imshow(image)
        print(label[index])
        plt.show()
        count +=1
        if count >= 5:
            break
    break
"""

steps_per_epoch = train_generator.n//train_generator.batch_size
steps_size_valid = valid_generator.n//valid_generator.batch_size


vgg = VGG16(weights="imagenet", include_top=False, input_shape=(100,100,3))
vgg.trainable = False
#vgg.summary()

set_trainable = False
for layer in vgg.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

vgg._get_trainable_state()

"""
model = models.Sequential()
model.add(vgg)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.25))
model.add(layers.Dense(15, activation='softmax'))


model.compile(optimizer=optimizers.RMSprop(lr=2e-5),loss="categorical_crossentropy", metrics=["acc"])
history = model.fit(train_generator, steps_per_epoch=steps_per_epoch, epochs=15, validation_data=valid_generator, validation_steps=steps_size_valid)
model.save("NN_VGG16.h5")

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc)+1)

plt.plot(epochs,acc,'bo', label='Training acc')
plt.plot(epochs, val_acc,'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
"""
