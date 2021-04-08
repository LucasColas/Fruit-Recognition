import numpy as np
#from tensorflow.keras import models, layers, optimizers, metrics
import os


main_path = r'E:\Projets code\DS Fruit recognition'
folders = os.listdir(main_path)



def get_data(path):
    folders_name = ["Train", "Validation"]
    folders = os.listdir(main_path)
    print(folders)

    for folder in folders:


get_data(main_path)
