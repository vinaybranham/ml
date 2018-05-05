# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 20:59:39 2018

@author: lekha
"""

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend
backend.set_image_dim_ordering('tf')

class NeuralNet:
    
    def build(self, no_of_items, width, height, depth, total_classes, Saved_Weights_Path=None):
        model = Sequential()

        model.add(Conv2D(140, 5, 5, border_mode="same", input_shape=(height, width, depth)))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering="tf"))
        
        model.add(Conv2D(120, 5, 5, border_mode="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering="tf"))
        
        model.add(Conv2D(100, 5, 5, border_mode="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering="tf"))
        
        model.add(Conv2D(100, 5, 5, border_mode="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering="tf"))
        
        model.add(Conv2D(100, 5, 5, border_mode="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering="tf"))
        
        model.add(Conv2D(90, 5, 5, border_mode="same"))
        model.add(Activation("relu"))
        #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering="tf"))
        
        model.add(Conv2D(90, 5, 5, border_mode="same"))
        model.add(Activation("relu"))
        #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering="tf"))
        
        model.add(Conv2D(80, 5, 5, border_mode="same"))
        model.add(Activation("relu"))
        #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering="tf"))
        
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation("relu"))

        model.add(Dense(total_classes))
        model.add(Activation("softmax"))

        if Saved_Weights_Path is not None:
            model.load_weights(Saved_Weights_Path)
        return model
        
