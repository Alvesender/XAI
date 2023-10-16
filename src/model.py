import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


def create_model(input_shape, conv_layer_num, filter_num, kernel_size, pool_size, 
                 dropout_coeff_conv, dense_layer_num, units_num, dropout_coeff_dense):
    # model = tf.keras.Sequential()
    
    # for i in range(conv_layer_num):
    #     model.add(tf.keras.layers.Conv2D(
    #         filters=filter_num, kernel_size=kernel_size, padding="same",
    #         activation='relu', input_shape = input_shape))
    #     model.add(tf.keras.layers.MaxPool2D(pool_size = pool_size, padding="same")) 
    #     model.add(tf.keras.layers.Dropout(dropout_coeff_conv))
    
    # #--------------------Fully-connected-----------------------------#
    # model.add(tf.keras.layers.Flatten())
    # layer_unit = model.count_params() // 10

    # print("layer unit(hidden):",layer_unit)
    # for i in range(dense_layer_num):
    #     model.add(tf.keras.layers.Dense(layer_unit, activation= 'relu'))
    #     model.add(tf.keras.layers.Dropout(dropout_coeff_dense))
    # model.add(tf.keras.layers.Dense(1,activation='sigmoid'))
    
    model = Sequential(
    [
        #Conv2D and MaxPool2D --> that used for images 
        #1D --> used for speach proccessing
        layers.Conv2D(128,9,padding='same',activation='relu',input_shape=(input_shape)),#image_size=200,,,,3=RGB-->if 1 =black and white image
        layers.MaxPool2D(),
        # layers.Dropout(0.50),#if model need this
     
        layers.Conv2D(64,6,padding='same',activation='relu',),
        layers.MaxPool2D(),
        # layers.Dropout(0.50),


        layers.Conv2D(32,3,padding='same',activation='relu',),
        layers.MaxPool2D(),
        # layers.Dropout(0.50),


        layers.Conv2D(16,3,padding='same',activation='relu',),
        layers.MaxPool2D(),
        # layers.Dropout(0.50),
        layers.Conv2D(8,1,padding='same',activation='relu',),
        layers.MaxPool2D(),
        # layers.Dropout(0.50),
     
        layers.Flatten(),
     
        layers.Dense(128 , activation='relu'),
        layers.Dense(128 , activation='relu'),
        layers.Dense(1 , activation='sigmoid'),#all layers with relu as activation expect the last layer , with smallest unit=3
    ]
    )
    
    optimizer = tf.keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimizer, loss = "binary_crossentropy",
                metrics=["accuracy"])

    return model