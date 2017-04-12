#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 15:15:58 2017

@author: avarfolomeev
"""

import keras
import csv
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
    

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
#limit amount of GPU memory per process

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.6
set_session(tf.Session(config=config))



#list of recordings on target computer
dataDirs = ['/media/D/DIZ/CarND/P3/tr21', '/media/D/DIZ/CarND/P3/tr22',
            '/media/D/DIZ/CarND/P3/tr23', 
            '/media/D/DIZ/CarND/P3/tr12',
            '/media/D/DIZ/CarND/P3/tr24', #(tr2 - reverse)
            '/media/D/DIZ/CarND/P3/tr25',
            '/media/D/DIZ/CarND/P3/tr10', '/media/D/DIZ/CarND/P3/tr11']

#modify path            
def newPath(file):
    # to use both Windows and Linux files:
    fname = file.replace("\\", "/").split('/')[-1];
    return dataDir + '/IMG/' + fname;


samples = []

for dataDir in dataDirs:
    csvName = dataDir + "/driving_log.csv";
    lines = [];

    with open(csvName) as csvFile:
        reader = csv.reader(csvFile);
        for line in reader:
            #replace original dir with the current
            for i in range(3):
                line[i] = newPath(line[i]);
    
            samples.append(line);

# NB! shuffle!
sklearn.utils.shuffle(samples)       


trn_smpl, val_smpl = train_test_split(samples, test_size=0.2);

#angle correction for left and right camera images
correction = 0.13; 
corrections = [0, correction, correction *-1.];


#generator function

def generator(samples, batchSize = 32):
    nSmpl = len(samples);
    
    while 1: #Forever
        sklearn.utils.shuffle(samples);
        for batchStart in range(0, nSmpl, batchSize):
            batchSamples = samples[batchStart:batchStart+batchSize];
    
            images = [];
            angles = [];
    
            for sample in batchSamples:
                centerAngle = float(sample[3])
                for i in range(3):
                    image = cv2.imread(sample[i]);
		  
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB);
                    crop = image[60:140,:]
                    cropf = (crop / 255.) - 0.5;
                    cropsm = cv2.resize(cropf,(160,40));
                    angle = centerAngle + corrections[i];
                    images.append(cropsm);
                    angles.append(angle);
                    images.append(cv2.flip(cropsm,1));
                    angles.append(angle * -1.);
            
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle (X_train, y_train)
      
trn_generator = generator(trn_smpl, batchSize = 64) 
val_generator = generator(val_smpl, batchSize=64)






#model construction

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization as BN
from keras.layers import Activation
from keras.layers.advanced_activations import PReLU
from keras.layers import Dropout

model_name='../model.h5'

model = Sequential()
model.add(Convolution2D(8,3,3,input_shape=(40, 160, 3))) #38x158x8
model.add(BN())
model.add(Activation('relu'))

model.add(Convolution2D(16,3,3)) #36x156x16
model.add(BN())
model.add(Activation('relu'))
model.add(MaxPooling2D())       #18x78x16


model.add(Convolution2D(24,3,3)) #16x76x24
model.add(BN())
model.add(PReLU())
model.add(MaxPooling2D())       #8x38x24


model.add(Convolution2D(32,3,3)) #6x36x32
model.add(BN())
model.add(PReLU())
model.add(MaxPooling2D())       #3x18x32

model.add(Convolution2D(64,3,3)) #1x16x64
model.add(BN())
model.add(PReLU())

model.add(Flatten())
model.add(Dense(100))
model.add(BN())
model.add(Dropout(0.5))
model.add(PReLU())
model.add(Dense(50))
model.add(PReLU())
model.add(Dropout(0.5))
model.add(Dense(1))

opt=keras.optimizers.adam(lr=2e-3, decay=1e-7)

model.compile(loss='mse', optimizer=opt, metrics = ['accuracy'])
model.summary()


model.fit_generator(trn_generator, 
                    samples_per_epoch = len(trn_smpl)*6,
                    validation_data = val_generator,
                    nb_val_samples=len(val_smpl)*6,
                    nb_epoch = 7);
                    
model.save(model_name)
