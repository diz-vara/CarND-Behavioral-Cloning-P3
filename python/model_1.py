#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 15:15:58 2017

@author: avarfolomeev
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 22:15:39 2017

@author: avarfolomeev
"""

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization as BN
from keras.layers.advanced_activations import PReLU
from keras.layers import Dropout
from keras.layers.convolutional import Cropping2D

model = Sequential()
model.add(Lambda(lambda x: (x / 255.) - 0.5, input_shape = (160,320,3)))
model.add(Cropping2D(cropping=((60,20),(0,0))))
model.add(Convolution2D(8,3,3)) #78x318x8
model.add(BN())
model.add(PReLU())

model.add(Convolution2D(12,3,3)) #76x316x12
model.add(BN())
model.add(PReLU())
model.add(MaxPooling2D())       #38x158x12


model.add(Convolution2D(16,3,3)) #36x156x16
model.add(BN())
model.add(PReLU())
model.add(MaxPooling2D())       #18x78x16


model.add(Convolution2D(24,3,3)) #16x76x24
model.add(BN())
model.add(PReLU())
model.add(MaxPooling2D())       #8x38x24

model.add(Convolution2D(32,3,3)) #6x36x32
model.add(BN())
model.add(PReLU())
model.add(MaxPooling2D())       #3x18x32


model.add(Convolution2D(48,3,3))  #1x16x48 
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

model.compile(loss='mse', optimizer='adam')

model.fit(X_train, y_train, validation_split=0.1, shuffle=True, nb_epoch=10)

model.save('../m1-.h5')
