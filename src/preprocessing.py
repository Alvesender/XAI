import cv2
import os
import numpy as np

def preprocess():
    src_folder_paths = [r'input/raw/Bike', r'input/raw/Car']
    target_folder_paths = [r'input/preprocessed/Bike', r'input/preprocessed/Car']
    
    for i in range(len(src_folder_paths)):
        for filename in os.listdir(src_folder_paths[i]):
            img = cv2.imread(os.path.join(src_folder_paths[i], filename))
            if img is not None:
                cv2.imwrite(os.path.join(target_folder_paths[i],filename), img)
                img = cv2.resize(img, (48, 48))
                cv2.imwrite(os.path.join(target_folder_paths[i],filename), img)

def load_prepocessed_data():
    folder_paths = [r'input/preprocessed/Bike', r'input/preprocessed/Car']
    
    # bike = 0, car = 1
    data = {0:[], 1:[]}
    
    for i, folder in enumerate(folder_paths):
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, filename))
            data[i].append(img)
    X = np.array(data[0] + data[1])
    y = np.array([0]*len(data[0]) + [1]*len(data[1]))

    return X, y

def load_raw_data():
    folder_paths = [r'input/raw/Bike', r'input/raw/Car']
    
    # bike = 0, car = 1
    data = {0:[], 1:[]}
    
    for i, folder in enumerate(folder_paths):
        for filename in os.listdir(folder):
            img = cv2.imread(os.path.join(folder, filename))
            data[i].append(img)
    X = data[0] + data[1]
    y = np.array([0]*len(data[0]) + [1]*len(data[1]))

    return X, y


if __name__ == '__main__':
    preprocess()