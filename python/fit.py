#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 11:50:35 2017

@author: avarfolomeev
"""

model.fit_generator(trn_generator, 
                    samples_per_epoch = len(trn_smpl)*6,
                    validation_data = val_generator,
                    nb_val_samples=len(val_smpl)*6,
                    nb_epoch = 7, initial_epoch=6);
                    