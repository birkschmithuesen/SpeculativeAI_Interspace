import keras
from keras.models import Sequential
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.optimizers import SGD
from keras.models import model_from_json
from keras.models import load_model
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

UDP_IP = 'localhost'
UDP_PORT=10005

MAKE_TRAINING = False


model = Sequential()

fft=[]

e1 = threading.Event()
e2 = threading.Event()


def fft_handler(unused_addr, args_1,args_2,args_3,args_4,args_5,args_6,args_7,args_8,args_9,args_10,args_11,args_12,args_13,args_14,args_15,args_16,args_17,args_18,args_19,args_20, args_21, args_22,args_23,args_24,args_25,args_26,args_27,args_28,args_29,args_30):
    print('received ftt paket. list-length: ',len(fft), '   first arg: ', args_1)
    fft.append(args_1)
    fft.append(args_2)
    fft.append(args_3)
    fft.append(args_4)
    fft.append(args_5)
    fft.append(args_6)
    fft.append(args_7)
    fft.append(args_8)
    fft.append(args_9)
    fft.append(args_10)
    fft.append(args_11)
    fft.append(args_12)
    fft.append(args_13)
    fft.append(args_14)
    fft.append(args_15)
    fft.append(args_16)
    fft.append(args_17)
    fft.append(args_18)
    fft.append(args_19)
    fft.append(args_20)
    fft.append(args_21)
    fft.append(args_22)
    fft.append(args_23)
    fft.append(args_24)
    fft.append(args_25)
    fft.append(args_26)
    fft.append(args_27)
    fft.append(args_28)
    fft.append(args_29)
    fft.append(args_30)
    e1.set()

frameCount=0;
def frameCountHandler(unused_addr, args):
    #print('received frameCoount: ', args)
    global frameCount
    frameCount=args;

def newModel_handler(unused_addr, args):
    kerasBackend.clear_session()
    my_init=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)

    model.add(Dense(6144, activation='sigmoid', input_dim=30, kernel_initializer=my_init, bias_initializer=my_init))
    model.add(Dense(13824, activation='sigmoid',kernel_initializer=my_init, bias_initializer=my_init))
    sgd = SGD(lr=0.06, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(train_X, train_y, epochs=1, batch_size=32, shuffle=True)
    model._make_predict_function()
    print('Loaded new model')


def train_handler(unused_addr, args):
    model.fit(train_X, train_y, epochs=10, batch_size=32, shuffle=True)
    model._make_predict_function()
    print('training finished...')
    print('')


def initialize_server():
  from pythonosc import dispatcher
  parser = argparse.ArgumentParser()
  parser.add_argument("--ip",
      default="0.0.0.0", help="The ip to listen on")
  parser.add_argument("--port",
      type=int, default=8000, help="The port to listen on")
  args = parser.parse_args()

  dispatcher = dispatcher.Dispatcher()
  dispatcher.map("/fft", fft_handler)
  dispatcher.map("/Playback/Recorder/frameCount", frameCountHandler)
  dispatcher.map("/train", train_handler)
  dispatcher.map("/newModel", newModel_handler)

  server = osc_server.ThreadingOSCUDPServer(
      (args.ip, args.port), dispatcher)
  server_thread=threading.Thread(target=server.serve_forever)
  server_thread.start()


def ledoutput():
    sock=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    e2.wait()
    while 0<1:
        e2.clear()
        #print('output buffer utilisation:',len(dout))
        #client.send_message("/Playback/Recorder/NNframeCount", frameCount)
        prediction_output=dout.pop(0)
        prediction_output=np.multiply(prediction_output,255)
        prediction_output=prediction_output.astype(np.uint8)
        for x in range(10):
          ledValues=prediction_output[(x*1402):((x+1)*1402):1]
          ledValues = ledValues - 127
          header=struct.pack('!IBB',frameCount,x,0)
          message=header+bytes(ledValues.tolist())
          sock.sendto(message, (UDP_IP, UDP_PORT))
        #wait till the next frame package is ready
        if(len(dout) < 1):
            e2.wait()


#load model from disk-------------------------------

#json_file=open('model_PERCEPTRON.json', 'r')
#model_json = json_file.read()
#json_file.close()
#model=model_from_json(model_json)
#print("model loaded")

#load weights into new model
#model.load_weights("model_PERCEPTRON.h5")
#print("weights loaded")

#end load model from disk-------------------------------

#TODO: the program will crash at this point if there is no network card found with the IP 2.0.0.1
initialize_server()

if MAKE_TRAINING:
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

    model.fit(train_X, train_y, epochs=30, batch_size=32, shuffle=True)
    model._make_predict_function()
    model.save("model.h5")
    print('Saved model')
else:
    model = keras.models.load_model("model.h5")
    print('Loaded saved model')

# we have two threads in the main loop:
# t1 is sending the visual output data via OSC to the JAVA program that is displaying it on the visual object
# t2 does the prediction
dout=[]
t1 = threading.Thread(name='ledoutput', target=ledoutput)
t1.start()
e1.wait()
while 0<1:
    e1.clear()
    prediction_input=np.empty([30,1])
    for x in range(30):
        prediction_input[x]=fft.pop(0) # get the fft data via OSC from Ableton Live
    prediction_input.shape=(1,30)
    prediction_output=model.predict(prediction_input)
    prediction_output=prediction_output.flatten()
    dout.append(prediction_output)
    e2.set()
    if len(fft)<30:
        e1.wait()
