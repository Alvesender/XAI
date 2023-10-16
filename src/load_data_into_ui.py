from sklearn.model_selection import train_test_split
from preprocessing import load_prepocessed_data, load_raw_data
import os
import cv2
import pandas as pd

import tensorflow as tf

def load_data_into_ui():
    X, y = load_raw_data()
    X_train_raw, X_test_raw, Y_train_raw, Y_test_raw = train_test_split(X, y, test_size=0.2, random_state=42)
    
    ui_paths = [r'dash_src/assets/images/raw/Car', r'dash_src/assets/images/raw/Bike']
    
    for ui_path in ui_paths:
        for filename in os.listdir(ui_path):
            file_path = os.path.join(ui_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    
    bike_counter = 0
    car_counter = 0
    for i in range(len(X_test_raw)):
        if Y_test_raw[i] == 0:
            cv2.imwrite(f'dash_src/assets/images/raw/Bike/Bike_{bike_counter}.png', X_test_raw[i])
            bike_counter += 1
        else:
            cv2.imwrite(f'dash_src/assets/images/raw/Car/Car_{car_counter}.png', X_test_raw[i])
            car_counter += 1
    
    
    X, y = load_prepocessed_data()
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    ui_paths = [r'dash_src/assets/images/preprocessed/Car', r'dash_src/assets/images/preprocessed/Bike']

    for ui_path in ui_paths:
        for filename in os.listdir(ui_path):
            file_path = os.path.join(ui_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    
    model = tf.keras.models.load_model('models/model.h5')
    y_pred = model.predict(X_test)
    
    bike_counter = 0
    car_counter = 0
    bike_pred = []
    car_pred = []
    for i in range(len(X_test)):
        if Y_test[i] == 0:
            cv2.imwrite(f'dash_src/assets/images/preprocessed/Bike/Bike_{bike_counter}.png', X_test[i])
            bike_counter += 1
            bike_pred.append(y_pred[i][0])
        else:
            cv2.imwrite(f'dash_src/assets/images/preprocessed/Car/Car_{car_counter}.png', X_test[i])
            car_counter += 1
            car_pred.append(y_pred[i][0])
            
    pd.DataFrame(bike_pred, columns=['pred_class']).to_csv('dash_src/assets/data/Bike_pred.csv')
    pd.DataFrame(car_pred, columns=['pred_class']).to_csv('dash_src/assets/data/Car_pred.csv')


if __name__ == '__main__':
    load_data_into_ui()