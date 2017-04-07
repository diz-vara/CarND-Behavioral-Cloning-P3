#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 10:18:34 2017

@author: avarfolomeev
"""

import keras
import csv
import cv2
from keras.backend import tf as ktf
    

dataDirs = ['/media/D/DIZ/CarND/P3/tr2', '/media/D/DIZ/CarND/P3/track2', '/media/D/DIZ/CarND/P3/tr21',
            '/media/D/DIZ/CarND/P3/t1', '/media/D/DIZ/CarND/P3/t2']
#'/media/D/DIZ/CarND/P3/track2' 

def newPath(file):
    fname = file.replace("\\", "/").split('/')[-1]
    return dataDir + '/IMG/' + fname;



ktf.GPUOptions(per_process_gpu_memory_fraction = 0.5)
        
images = []
angles = []

correction = 0.13 # this is a parameter to tune

for dataDir in dataDirs:
    csvName = dataDir + "/driving_log.csv";
    lines = []

    with open(csvName) as csvFile:
        reader = csv.reader(csvFile);
        for line in reader:
            lines.append(line)

    for line in lines:
        for i in range(3):
            image = cv2.imread(newPath(line[i]))
            image = cv2.resize(image,(160,80))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images.append(image)
        
        angle = float(line[3])
        angle_left = angle + correction
        angle_right = angle - correction
        angles.extend([angle, angle_left, angle_right])

#X_train = np.array(images)
#y_train = np.array(angles)
