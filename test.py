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
        image = cv2.imread(os.path.join(path_class, image))
        new_image = cv2.resize(image, (100,100))
        test_images.append(image)
        break


np_images = np.asarray(test_images).reshape(-1,100,100,3)
