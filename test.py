#from tensorflow.keras import models
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from tensorflow.keras import models

test_path = r'E:\Projets code\Dataset Fruit Recognition\Test'


classes = os.listdir(test_path)
test_images = {}

for i,classe in enumerate(classes):
    path_class = os.path.join(test_path, classe)
    images = os.listdir(path_class)
    count = 1
    for image in images:
        try:

            img = cv2.imdecode(np.fromfile(os.path.join(path_class, image), dtype=np.uint8), cv2.IMREAD_UNCHANGED)

            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            new_image = cv2.resize(rgb_image, (100,100))
            test_images[classe] = new_image

            print("classe : ", classe)
            if count >= 1:
                break
            #print(count)
            count += 1


        except Exception as e:
            print("error", str(e))


model = models.load_model("NN_VGG16.h5")


def prediction(images):
    for image in images.items():
        print(image[1])
        x = np.array(image[1], dtype="float32").reshape(-1,100,100,3)
        x /= 255

        predict = model.predict(x)
        print(image[0])
        print(predict.shape)
        print(np.argmax(predict))

        plt.clf()
        plt.imshow(image[1])
        plt.show()

prediction(test_images)
