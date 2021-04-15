#from tensorflow.keras import models
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from tensorflow.keras import models

test_path = r'E:\Projets code\Dataset Fruit Recognition\Test'


classes = os.listdir(test_path)
test_images = []

for i,classe in enumerate(classes):
    path_class = os.path.join(test_path, classe)
    images = os.listdir(path_class)
    count = 1
    for image in images:
        try:

            img = cv2.imdecode(np.fromfile(os.path.join(path_class, image), dtype=np.uint8), cv2.IMREAD_UNCHANGED)

            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            new_image = cv2.resize(rgb_image, (100,100))
            test_images.append(new_image)
            print("classe : ", classe)
            if count >= 1:
                break
            #print(count)
            count += 1


        except Exception as e:
            print("error", str(e))





np_images = np.asarray(test_images).reshape(-1,100,100,3)
#print(np_images)
#print(np_images.shape)


model = models.load_model("NN_VGG16.h5")


def prediction(images):
    for image in images:
        x = np.array(image, dtype="float32").reshape(-1,100,100,3)
        x //= 255

        predict = model.predict(x)

        print(predict.shape)
        print(np.argmax(predict))

        plt.clf()
        plt.imshow(image)
        plt.show()

prediction(test_images)
