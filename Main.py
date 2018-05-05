# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 18:29:24 2018

@author: isara
"""
from temp import NeuralNet
import test
from test import Dataset
import numpy as np
from  keras.models import model_from_json
import os.path

class main:
    def __init__(self):
        self.model = None
        self.ds = None
    def train(self):
        #path = "E:/Masterzz/6156-MachineLearning/Project/extracted_images"
        
        if(not (os.path.exists('model.json')) and not (os.path.exists('model.h5'))):
            imgPath = "E:/Masterzz/6156-MachineLearning/Project/test2"
            self.ds = Dataset(imgPath)
            dataset,labels,total_classes,size = self.ds.Extraction()
            dataset = np.expand_dims(dataset, axis=3)
            print("extracted")
            Nn = NeuralNet()
            self.model = Nn.build(size,45,45,1,total_classes)
            self.model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['categorical_accuracy'])
            self.model.fit(dataset.reshape(dataset.shape[0], 45, 45, 1),labels, batch_size=64,epochs = 1)
        #scores = model.evaluate(dataset, labels)
            print(self.model.metrics_names[1])
            self.model_json = self.model.to_json()
            with open("model.json", "w") as json_file:
                json_file.write(self.model_json)
            self.model.save_weights("model.h5")
        else:
            json_file = open("model.json", "r")
            model_json = json_file.read()
            json_file.close()
            self.model = model_from_json(model_json)
            self.model.load_weights('model.h5')
        #model.fit(dataset,labels, batch_size=16,epochs = 10)
    def test(self,imgList):
        imgPredict=self.model.predict(imgList)
        return imgPredict

'''
ds = Dataset("E:/Masterzz/6156-MachineLearning/Project/test")
dataset,labels,total_classes,size = ds.Extraction()
dataset.shape
import numpy as np
image = np.expand_dims(dataset, axis=3)
image.shape
dataset[0,0,0]
image[0,0,0,:]
'''