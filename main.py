import numpy as np
#from tensorflow.keras import models, layers, optimizers, metrics
import os


main_path = r'E:\Projets code\Fruit recognition'
folders = os.listdir(main_path)

for folder in folders:
    print(folder)
    files, extension = os.path.splitext(folder)
    print(files, extension)
