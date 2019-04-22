import keras
from keras.models import Sequential
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.optimizers import SGD
from keras.models import model_from_json
from keras import backend as kerasBackend

import argparse
import random
import time
import math
import threading
import socket
import struct

from pythonosc import udp_client
from pythonosc import osc_server
from multiprocessing import Process, Queue

import numpy as np

import os
from numpy import loadtxt

model = Sequential()

#import fft and led input data
file_name='traingsdata.txt'
file = open(file_name)
print('Loading Trainingsdata from File:', file_name,'  ...')
values=loadtxt(file_name, dtype='float32')
print('Trainingsdata points: ',values.shape[0])
print()

#split into input and outputs
train_X, train_y = values[:,:-13824], values[:,30:]
print('Train_X: ', train_X.shape, 'Train_y: ', train_y.shape)

# LOAD NN MODEL
my_init=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
model.add(Dense(6144, activation='sigmoid', input_dim=30, kernel_initializer=my_init, bias_initializer=my_init))
model.add(Dense(13824, activation='sigmoid',kernel_initializer=my_init, bias_initializer=my_init))

sgd = SGD(lr=0.06, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(train_X, train_y, epochs=10000, batch_size=605, shuffle=True)
model.save('model.h5')
print('Saved new model 2 disk')
