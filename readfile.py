from utils import mnist_reader
import matplotlib.pyplot as plt
import random
import numpy as np
from PIL import Image
import os

X_train, y_train = mnist_reader.load_mnist('data', kind='train')
X_test, y_test = mnist_reader.load_mnist('data', kind='t10k')

imgH,imgW=28,28

dataPath='data/'
trainPath='train/'
testPath='test/'

## for train data printing
if not os.path.exists(dataPath+trainPath):
    os.mkdir(dataPath+trainPath)

for i in range(len(X_train)):
    tempImg=X_train[i]
    tempLabel=y_train[i]
    tempImg=np.array(tempImg)
    tempImg=tempImg.reshape((imgH,imgW))
    tempImg=Image.fromarray(tempImg)
    tempName=str(tempLabel)+'_'+str(i)+'.jpg'
    tempPath=dataPath+trainPath+str(tempLabel)+'/'
    if not os.path.exists(tempPath):
        os.mkdir(tempPath)
    tempImg.save(tempPath+tempName)
    if i%1000==0:
        print(str(i)+'/'+str(len(X_train))+' data done')

print('train data get done!')

## for test data printing
if not os.path.exists(dataPath+testPath):
    os.mkdir(dataPath+testPath)

for i in range(len(X_test)):
    tempImg=X_test[i]
    tempLabel=y_test[i]
    tempImg=np.array(tempImg)
    tempImg=tempImg.reshape((imgH,imgW))
    tempImg=Image.fromarray(tempImg)
    tempName=str(tempLabel)+'_'+str(i)+'.jpg'
    tempPath=dataPath+testPath+str(tempLabel)+'/'
    if not os.path.exists(tempPath):
        os.mkdir(tempPath)
    tempImg.save(tempPath+tempName)
    if i%1000==0:
        print(str(i)+'/'+str(len(X_test))+' data done')

print('test data get done!')
