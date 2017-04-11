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

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8
set_session(tf.Session(config=config))


from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import AveragePooling2D
from keras.layers import BatchNormalization as BN
from keras.layers import Activation
from keras.layers.advanced_activations import PReLU
from keras.layers import Dropout

model_name='../m1-gnR.h5'

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

opt=keras.optimizers.adam(lr=5e-3, decay=1e-7)

model.compile(loss='mse', optimizer=opt, metrics = ['accuracy'])
model.summary()


model.fit_generator(trn_generator, 
                    samples_per_epoch = len(trn_smpl)*6,
                    validation_data = val_generator,
                    nb_val_samples=len(val_smpl)*6,
                    nb_epoch = 7);
                    
model.save(model_name)
