import numpy as np
#from tensorflow.keras import models, layers, optimizers, metrics
import os


main_path = r'E:\Projets code\DS Fruit recognition'
folders = os.listdir(main_path)


folders_name = ["Train", "Validation"]
def get_data(path, folder):

    classes_path = os.path.join(path, folder)
    classes = os.listdir(classes_path)
    print(classes)
    for classe in folders_name:
        pass



X_train = get_data(main_path, folders_name[0])
X_Val = None
