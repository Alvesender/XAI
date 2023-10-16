import tensorflow as tf
from model import create_model

from sklearn.model_selection import train_test_split
from scikeras.wrappers import KerasClassifier
from preprocessing import load_prepocessed_data
from sklearn.preprocessing import OneHotEncoder
import os

import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import os

def train():
    
    X, y = load_prepocessed_data()
    # y = OneHotEncoder().fit_transform(y.reshape(-1, 1)).toarray()
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = create_model(
        input_shape = X_train.shape[1:],
        conv_layer_num=32, 
        filter_num=3, 
        kernel_size=(8,8),
        pool_size = (2,2), 
        dropout_coeff_conv=0.2, 
        dense_layer_num=2, 
        units_num = 128, 
        dropout_coeff_dense = 0.5
    )
    model.fit(X_train, Y_train, epochs=10, validation_data=(X_test, Y_test))
    model.save('models/model.h5')

if __name__ == '__main__':
    train()
    