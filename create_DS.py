import os
import cv2
import matplotlib.pyplot as plt

main_path = r'E:\Projets code\DS Fruit recognition'
folders = os.listdir(main_path)


folders_name = ["Train", "Validation"]
def get_data(path, folder):
    data = []
    classes_path = os.path.join(path, folder)
    #print(classes_path)
    classes = os.listdir(classes_path)
    print(classes)


    for classe in classes:
        path_images = os.path.join(classes_path, classe)
        images = os.listdir(path_images)
        #print(len(images))

        for image in images:
            label_one_hot = [0 for j in range(len(classes))]
            img = cv2.imread(os.path.join(path_images,image))
            img_resize = cv2.resize(img, (120,120))
            data.append()




data_train = get_data(main_path, folders_name[0])
X_Val = None
