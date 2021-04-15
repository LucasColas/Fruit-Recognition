#from tensorflow.keras import models
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2

test_path = r'E:\Projets code\Dataset Fruit Recognition\Test'


classes = os.listdir(test_path)
test_images = []

for classe in classes:
    path_class = os.path.join(test_path, classe)
    images = os.listdir(path_class)
    for image in images:
        try:

            cv2.imdecode(np.fromfile(os.path.join(path_class, image), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            print(img)
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            new_image = cv2.resize(rgb_image, (100,100))
            test_images.append(new_image)
            print("classe : ", classe)
            break

        except Exception as e:
            print("error", str(e))



np_images = np.asarray(test_images).reshape(-1,100,100,3)
#print(np_images)
print(np_images.shape)
