# chess
```py
%matplotlib inline
import random
import numpy as np 
import pandas as pd 
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Activation, Conv2D, MaxPooling2D

train_King = "King"
path_King = os.path.join(train_King)

train_Knight = "Knight"
path_Knight = os.path.join(train_Knight)

X = []
y = []

convert = lambda category : int(category == 'King')

def create_test_data(path):
    for i in os.listdir(path):        #all item in train
        category = os.path.join(path)    #file name
        category = convert(category)  #King or Knight
        
        #input image and grayscale
        img_array = cv2.imread(os.path.join(path,i),cv2.IMREAD_GRAYSCALE)
        
        #resize
        new_img_array = cv2.resize(img_array, dsize=(80, 80))
        
        X.append(new_img_array)
        y.append(category)
        
create_test_data(path_King)
create_test_data(path_Knight)
X = np.array(X).reshape(-1, 80,80,1)
y = np.array(y)

#Normalize data
X = X/255.0

model = Sequential()

model.add(Conv2D(32, (5,5),  input_shape = X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (5,5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128, (5,5)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(10))
model.add(Activation('sigmoid'))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer="adam",
              loss='binary_crossentropy',
              metrics=['accuracy'])
              
model.fit(X, y, epochs=20, batch_size=50)              
```
