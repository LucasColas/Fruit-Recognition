import numpy as np
#from tensorflow.keras import models, layers, optimizers, metrics
import os


main_path = r'E:\Projets code\Fruit recognition'
folders = os.listdir(main_path)


for folder in folders:
    #print(os.path.join(main_path, folder))
    files = os.listdir(os.path.join(main_path, folder))
    if len(files) < 8:
        for subfolder in files:
            print(os.path.join(main_path, folder, subfolder))
            files = os.listdir(os.path.join(main_path, folder, subfolder))
