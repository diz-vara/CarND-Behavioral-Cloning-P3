#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 10:18:34 2017

@author: avarfolomeev
"""

import keras
import csv
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
    

dataDirs = ['/media/D/DIZ/CarND/P3/tr21', '/media/D/DIZ/CarND/P3/tr22',
            '/media/D/DIZ/CarND/P3/tr23', 
            '/media/D/DIZ/CarND/P3/tr12',
            '/media/D/DIZ/CarND/P3/tr24', #(tr2 - reverse)
            '/media/D/DIZ/CarND/P3/tr25',
            '/media/D/DIZ/CarND/P3/tr10', '/media/D/DIZ/CarND/P3/tr11']

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

sklearn.utils.shuffle(samples)       
trn_smpl, val_smpl = train_test_split(samples, test_size=0.2);

correction = 0.13; 
corrections = [0, correction, correction *-1.];

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
      
ch, row, col = 3, 40, 160       
trn_generator = generator(trn_smpl, batchSize = 64) 
val_generator = generator(val_smpl, batchSize=64)

           