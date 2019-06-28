"""
This program makes predictions for audio input to visualize on the object Interspace.
The computation process is structured as followed:
0. trainingsdata + model: loading the trainingsdata from a textfile. Convert the
    trainingpoints in the right dimension: One line in the text file is one trainingpoint
    with 30 FFT values and 13824 LED values, separated by tabulators
1. when receiving the the OSC command "/train" the function train_handler executes model.fit
2. input: receive audio
3. FFT analysis: separate continues audio stream into 30 frequency bands with a frequency of 30fps
4. prediction: convert the FFT analyses results to feed into a neural network.
    Predict predict the brightness values of the 13824 Leds of the Interspace object
"""

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

# the ip and port to send the LED data to. The program Ortlicht receives them via OSC and
# converts them to ArtNet
UDP_IP = 'localhost'
UDP_PORT = 10005

model = Sequential()

INPUT_DIM = 30
NUM_SOUNDS = 60  # 2 seconds and 30 sounds per second?
LSTM_OUT = 512
BATCH_SIZE = 32
EPOCHS = 32

HIDDEN1_DIM = 1024
OUTPUT_DIM = 13824
# input shape is (?, NUM_SOUNDS, INPUT_DIM): ? is the batch size, NUM_SOUNDS is the number
#                                            of sounds in the sequence and INPUT_DIM is
#                                            the vector size of each sound
# output_shape is (?, LSTM_OUT): ? is the batch size and LSTM_OUT is the number of
# units in the output
# 'return_sequences' is False because you want only to codify NUM_SOUNDS sounds in
# one LSTM_OUT-dimensional vector


fft=[]

e1 = threading.Event()
e2 = threading.Event()

def fft_handler(*args):
    """
    this function is called when fft values are received via OSC (from ableton Live)
    """
    print('received ftt paket. list-length: ', len(fft), '   first arg: ', args[1])
    for arg in args[1:]:
        fft.append(arg)
    e1.set()

frameCount = 0
def frameCountHandler(unused_addr, args):
    """
    a function to synchronize for recording the output of the neural network
    """
    #print('received frameCoount: ', args)
    global frameCount
    frameCount = args

def newModel_handler(unused_addr, args):
    """
    this function should reinitialize the model, to start the training from scratch again.
    ToDo: make it work. Probably the crash is caused because of the multi-threading?
    """
    kerasBackend.clear_session()
    my_init = keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)

    model.add(Dense(6144, activation='sigmoid', input_dim=30, kernel_initializer=my_init,
                    bias_initializer=my_init))
    model.add(Dense(13824, activation='sigmoid',kernel_initializer=my_init,
                    bias_initializer=my_init))
    sgd = SGD(lr=0.06, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(training_input, training_output, epochs=1, batch_size=32, shuffle=True)
    model._make_predict_function()
    print('Loaded new model')

def train_handler(unused_addr, args):
    """
    neural network trainer
    """
    model.fit(training_input, training_output, epochs=EPOCHS, batch_size=BATCH_SIZE, shuffle=True)
    model._make_predict_function()
    print('training finished...')
    print('')

def initialize_server():
    """
    the  OSC server
    """
    from pythonosc import dispatcher
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="0.0.0.0", help="The ip to listen on")
    parser.add_argument("--port", type=int, default=8000, help="The port to listen on")
    args = parser.parse_args()
    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("/fft", fft_handler)
    dispatcher.map("/Playback/Recorder/frameCount", frameCountHandler)
    dispatcher.map("/train", train_handler)
    dispatcher.map("/newModel", newModel_handler)
    try:
        server = osc_server.ThreadingOSCUDPServer((args.ip, args.port), dispatcher)
        server_thread=threading.Thread(target=server.serve_forever)
        server_thread.start()
    except OSError as e:
        server = None
        server_thread = None
        print(e)


def ledoutput():
    """
    runs parallel in a single thread
    when a new prediction is ready, it sends the LED data via OSC to 'Ortlicht'
    """
    sock=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    e2.wait()
    while True:
        e2.clear()
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

initialize_server()

#import fft and led input data
file_name='traingsdata.txt'
file = open(file_name)
print('Loading Trainingsdata from File:', file_name,'  ...')
values=loadtxt(file_name, dtype='float32')
print('Trainingsdata points: ',values.shape[0])
print()

batch_size = (len(values)+1)/NUM_SOUNDS
batch_size = int(batch_size)
print("Batch size:" + str(batch_size))
training_input = np.empty([batch_size, NUM_SOUNDS, INPUT_DIM])
training_output = np.empty([batch_size, OUTPUT_DIM])

input_batch = np.empty([NUM_SOUNDS, INPUT_DIM])
output_batch = np.empty([NUM_SOUNDS, OUTPUT_DIM])
batch_counter = 0
# split up input rows into batches and seperate input and outputs in the values array
for counter, row in enumerate(values):
    if counter % NUM_SOUNDS is 0 and counter > 0:
        training_input[batch_counter], training_output[batch_counter] = input_batch.copy(), output_batch.copy()
        batch_counter += 1
    input_batch[counter % NUM_SOUNDS] = values[counter,:-OUTPUT_DIM]
    output_batch = values[counter,INPUT_DIM:]


print('training_input shape: ', training_input.shape, 'training_output shape: ', training_output.shape)

"""
Initialize NeuralNetwork with LSTM.
ToDo: Make the input vector dimensions fit the LSTM (NUM_SOUNDS!)
"""
my_init=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)

model.add(keras.layers.LSTM(units=LSTM_OUT, input_shape=(NUM_SOUNDS, INPUT_DIM),
                            return_sequences=False, name='lstm_layer'))

# This is a hidden layer. You can use it or not.
# In this case the activation can be ReLU. I write down 2048 output units but you can try other quantities
model.add(keras.layers.Dense(units=HIDDEN1_DIM, activation='relu', name='hidden1_layer', kernel_initializer=my_init, bias_initializer=my_init))

# the output layer must have 13000 units (one per led) and the activation has to be sigmoid
model.add(keras.layers.Dense(units=OUTPUT_DIM, activation='sigmoid', name='output_layer', kernel_initializer=my_init, bias_initializer=my_init))

# define the optimizer. You can use the optimizer that you want
adam = keras.optimizers.Adam(lr=0.0001)

# and finally use binary_crossentropy as loos function
# model.compile(loss='binary_crossentropy', optimizer=adam)
sgd = SGD(lr=0.06, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.summary()
model.fit(training_input, training_output, epochs=1, batch_size=BATCH_SIZE, shuffle=True)
model._make_predict_function()
print('Loaded new model')


"""
main loop:
we have two threads:
t1 is sending the visual output data via OSC to the JAVA program that is displaying it on the visual object
t2 does the prediction
"""
dout=[]
t1 = threading.Thread(name='ledoutput', target=ledoutput)
t1.start()
e1.wait()
while 0<1:
    e1.clear()
    prediction_input=np.empty([INPUT_DIM,NUM_SOUNDS])
    for i in range(NUM_SOUNDS):
        e1.clear()
        for x in range(INPUT_DIM):
            prediction_input[x,i]=fft.pop(0) # get the fft data via OSC from Ableton Live
        if len(fft)<30:
            e1.wait()
    e1.clear()
    #prediction_input = np.reshape=(prediction_input, (1,NUM_SOUNDS,INPUT_DIM))
    prediction_input.shape=(1,NUM_SOUNDS,INPUT_DIM)
    prediction_output=model.predict(prediction_input)
    prediction_output=prediction_output.flatten()
    print("Finished prediction with shape:" + str(prediction_output.shape))
    print(prediction_output)
    dout.append(prediction_output)
    e2.set()
    if len(fft)<30:
        e1.wait()
