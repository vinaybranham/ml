# -*- coding: utf-8 -*-
"""
Created on Sat Mar 31 17:51:40 2018

@author: isara
"""
import os
import numpy as np
from PIL import Image
from pylab import array
from keras.utils import np_utils
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Dataset:
    def __init__(self,a):
        self.path = a
        self.symbols = None
    
    def Extraction(self):
        cur = os.getcwd()
        datadir = self.path
        self.symbols = os.listdir(datadir)
        
        total_classes = len(self.symbols)
        size = 0
        
        for dir in self.symbols:
            images = os.listdir(datadir + '/' + dir)
            size += len(images)
        
        dataset = np.ndarray((size, 45, 45))
        labels = np.ndarray((size,1))
        i = 0
        for dir in self.symbols:
            images = os.listdir(datadir + '/' + dir)
            for image in images:
                im = Image.open(datadir + '/' + dir + '/' + image, 'r')
                pix_val = array(im.getdata())
                dataset[i] = pix_val.reshape((45,45))
                labels[i] = self.symbols.index(dir)
                
                i += 1
        labels = np_utils.to_categorical(labels, total_classes)
        #print(labels)
        return dataset,labels,total_classes,size
    

    
