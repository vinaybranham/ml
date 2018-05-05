# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 19:00:49 2018

@author: vinay
"""

import cv2
import numpy as np
import os
from test import Dataset
import traceback
from PIL import Image
from pylab import array

cur = os.getcwd()
datadir = "E:/Masterzz/6156-MachineLearning/Project/test4"
symbols = os.listdir(datadir)

size = 0

for dir in symbols:
    images = os.listdir(datadir + '/' + dir)
    size += len(images)

dataset = np.ndarray((size, 45, 45))
labels = np.ndarray((size,1))
i = 0
path = "E:/Masterzz/6156-MachineLearning/Project/test3"
kernel = np.ones((5,5), np.uint8)
for dir in symbols:
    images = os.listdir(datadir + '/' + dir)
    for image in images:
        img = cv2.imread(datadir + '/' + dir + '/' + image)
        imblur = cv2.GaussianBlur(img, (5,5), 0)
        _, thresh = cv2.threshold(imblur,200,255,cv2.THRESH_BINARY)
        imblur = cv2.GaussianBlur(thresh, (3,3), 5)
        _, thresh = cv2.threshold(imblur,150,255,cv2.THRESH_BINARY)
        
        imgArr = cv2.resize(thresh[:,:,1], (45,45))
        cv2.imwrite(path+ '/' + dir + '/' + image, imgArr)    
        
        i += 1

#print(pix_val.shape)
'''
for i in range(90520,94257):
    imgArr = dataset[i,:]
    #erosion = cv2.erode(imgArr,kernel,iterations = 1)
    path = "E:/Masterzz/6156-MachineLearning/Project/test1/9/"
    cv2.imwrite(path+str(i)+'.jpg', imgArr)

dataset.shape
'''
'''
try:
    img = cv2.imread("E:/Masterzz/6156-MachineLearning/Project/test1/6/81428.jpg")
    print(img.shape)
    cv2.namedWindow('test', cv2.WINDOW_NORMAL)
    kernel = np.ones((5,5), np.uint8)
    imblur = cv2.GaussianBlur(img, (5,5), 0)
    #ret,thresh = cv2.threshold(imblur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, thresh = cv2.threshold(imblur,200,255,cv2.THRESH_BINARY)
    #erosion = cv2.erode(dataset[0,:],kernel,iterations = 1)
    imblur = cv2.GaussianBlur(thresh, (3,3), 5)
    _, thresh = cv2.threshold(imblur,150,255,cv2.THRESH_BINARY)
    cv2.imshow('test', thresh[:,:,1])
    print(thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
except:
    print(traceback.print_exc())
    cv2.destroyAllWindows()
'''