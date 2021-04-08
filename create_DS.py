import os
import cv2
import matplotlib.pyplot as plt

main_path = r'E:\Projets code\DS Fruit recognition'
folders = os.listdir(main_path)


folders_name = ["Train", "Validation", "Test"]
def get_data(path, folder):
    data = []
    classes_path = os.path.join(path, folder)
    #print(classes_path)
    classes = os.listdir(classes_path)
    print(classes)


    for i,classe in enumerate(classes):
        path_images = os.path.join(classes_path, classe)
        images = os.listdir(path_images)
        #print(len(images))
        count = 0
        for image in images:
            label_one_hot = [0 for j in range(len(classes))]
            try:
                img = cv2.imread(os.path.join(path_images,image))
                img_resize = cv2.resize(img, (120,120))
                label_one_hot[i] = 1
                data.append((img_resize, label_one_hot))
                print(data)
            except  Exception as e:
                print(str(e))


def get_X_y(data):
    X_train = []
    y_train = []
    for piece_data, label in data:
        X_train.append(piece_data)
        y_train.append(label)

    return (X_train, y_train)

data_train = get_data(main_path, folders_name[0])
X_train, y_train = get_X_y(data)

data_val = get_data(main_path, folders_name[1])
X_val, y_val = get_X_y(data_val)
