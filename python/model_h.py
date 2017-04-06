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
from keras.backend import tf as ktf

model = Sequential()
model.add(Lambda(lambda x: (x / 255.) - 0.5, input_shape = (160,320,3)))
model.add(Lambda(lambda img: ktf.image.resize_bilinear(img,(80,160) ) ) )
model.add(Cropping2D(cropping=((30,10),(0,0))))
model.add(Convolution2D(8,3,3))
model.add(BN())
model.add(PReLU())
model.add(Convolution2D(32,3,3))
model.add(BN())
model.add(PReLU())
model.add(MaxPooling2D())
model.add(Convolution2D(32,3,3))
model.add(BN())
model.add(PReLU())
model.add(Convolution2D(64,3,3))
model.add(BN())
model.add(PReLU())
model.add(MaxPooling2D((1,2)))
model.add(Convolution2D(64,3,3))
model.add(BN())
model.add(PReLU())
model.add(MaxPooling2D())
model.add(Convolution2D(64,3,3))
model.add(BN())
model.add(Flatten())
model.add(Dense(80))
model.add(BN())
model.add(Dropout(0.5))
model.add(PReLU())
model.add(Dense(60))
model.add(PReLU())
#model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

#model.fit(X_train, y_train, validation_split=0.1, shuffle=True, nb_epoch=10)

#model.save('m1_h.h5')
