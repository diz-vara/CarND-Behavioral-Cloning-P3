#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 23:04:38 2017

@author: avarfolomeev
"""
import matplotlib.pyplot as plt
import numpy as np

a_images, a_angles = [],[]

for image, angle in zip(images, angles):
    a_images.append(image)
    a_angles.append(angle)
    a_images.append(cv2.flip(image,1))
    a_angles.append(angle * -1.)
    
#%%    
X_train = np.array(a_images)
y_train = np.array(a_angles)
    