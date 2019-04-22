import keras
from keras.models import Sequential
from keras.utils import plot_model
from keras.utils.vis_utils import model_to_dot
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.optimizers import SGD
from keras.models import model_from_json
from keras import backend as kerasBackend
from keras.models import load_model

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

UDP_IP = '2.0.0.2'
UDP_PORT=10005

model = Sequential()

fft=[]

e1 = threading.Event()
e2 = threading.Event()


def fft_handler(*args):
    print('received ftt paket. list-length: ',len(fft), '   first arg: ', args[1])
    for arg in args[1:]:
      fft.append(arg)
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
    model.fit(train_X, train_y, epochs=800, batch_size=32, shuffle=True)
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

  try:
      server = osc_server.ThreadingOSCUDPServer(
        (args.ip, args.port), dispatcher)
      server_thread=threading.Thread(target=server.serve_forever)
      server_thread.start()
  except OSError as e:
      server = None
      server_thread = None
      print(e)


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
model=load_model('model.h5')
model._make_predict_function()
print('Loaded new model')

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
