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
import random
from collections import deque
from numpy import loadtxt
from fft import SpectrumAnalyzer, FPS

# the ip and port to send the LED data to. The program Ortlicht receives them via OSC and
# converts them to ArtNet
UDP_IP = '2.0.0.2'
UDP_PORT = 10005
OSC_LISTEN_IP = "0.0.0.0" # =>listening from any IP
OSC_LISTEN_PORT = 8000

LOAD_MODEL = True
SAVE_MODEL = False

model = Sequential()

PREDICTION_BUFFER_MAXLEN = 441 # 10 seconds * 44.1 fps
PAUSE_LENGTH = 88 # length in frames of silence that triggers pause event
PAUSE_SILENCE_THRESH = 50 # Threshhold defining pause if sum(fft) is below the value
MIN_FRAME_REPLAYS = 0 # set the minimum times, how often a frame will be written into the buffer
MAX_FRAME_REPLAYS = 2 # set the maximum times, how often a frame will be written into the buffer

INPUT_DIM = 128
NUM_SOUNDS = 1
BATCH_SIZE = 32
EPOCHS = 30
INITIAL_EPOCHS = 500

HIDDEN1_DIM = 512
HIDDEN2_DIM = 4096
OUTPUT_DIM = 13824



fft = []
prediction_buffer = deque(maxlen=PREDICTION_BUFFER_MAXLEN)
e1 = threading.Semaphore(0)
pause_event = threading.Event()
pause_counter = 0

def fft_callback_function(fft_data):
    """
    this function is called when fft values are received via OSC (from ableton Live)
    """
    fft.append(list(fft_data))
    e1.release()

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
    parser.add_argument("--ip", default=OSC_LISTEN_IP, help="The ip to listen on")
    parser.add_argument("--port", type=int, default=OSC_LISTEN_PORT, help="The port to listen on")
    args = parser.parse_args()
    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("/Playback/Recorder/frameCount", frameCountHandler)
    dispatcher.map("/train", train_handler)
    dispatcher.map("/newModel", newModel_handler)
    try:
        server = osc_server.ThreadingOSCUDPServer((args.ip, args.port), dispatcher)
        server_thread=threading.Thread(target=server.serve_forever, daemon=True)
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
    pause_event.wait()
    while True:
        pause_event.clear()
        while len(prediction_buffer) > 0:
            prediction_output = prediction_buffer.popleft()
            prediction_output=np.multiply(prediction_output,255)
            prediction_output=prediction_output.astype(np.uint8)
            for x in range(10):
                ledValues=prediction_output[(x*1402):((x+1)*1402):1]
                header=struct.pack('!IBB',frameCount,x,0)
                message=header+bytes(ledValues.tolist())
                sock.sendto(message, (UDP_IP, UDP_PORT))
            time.sleep(1/FPS) #ensure playback speed matches framerate
        #wait till the next frame package is ready
        pause_event.wait()

def is_pause(fft_data):
    """
    return True if a pause in the fft stream was detected else return False
    """
    global pause_counter
    fft_sum = 0
    for fft_frame in fft_data:
        fft_sum += sum(fft_frame)
    if fft_sum < PAUSE_SILENCE_THRESH:
        if pause_counter >= PAUSE_LENGTH:
            pause_counter = 0
            return True
        pause_counter += 1
    else:
        pause_counter = 0
    return False


initialize_server()

"""
Initialize NeuralNetwork.
"""

if LOAD_MODEL:
    model=load_model('model.h5')
    model._make_predict_function()
    print('Loaded saved model from file')
else:
    #import fft and led input data
    file_name='traingsdata.txt'
    file = open(file_name)
    print('Loading Trainingsdata from File:', file_name,'  ...')
    values=loadtxt(file_name, dtype='float32')
    print('Trainingsdata points: ',values.shape[0])
    print()

    #split into input and outputs
    training_input, training_output = values[:,:-13824], values[:,INPUT_DIM:]
    print('training_input shape: ', training_input.shape, 'training_output shape: ', training_output.shape)
    my_init=keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None)
    model.add(Dense(HIDDEN1_DIM, activation='sigmoid', input_dim=INPUT_DIM, kernel_initializer=my_init, bias_initializer=my_init))
    model.add(Dense(HIDDEN2_DIM, activation='sigmoid', input_dim=HIDDEN1_DIM, kernel_initializer=my_init, bias_initializer=my_init))
    model.add(Dense(OUTPUT_DIM, activation='sigmoid',kernel_initializer=my_init, bias_initializer=my_init))
    sgd = SGD(lr=0.06, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(training_input, training_output, epochs=INITIAL_EPOCHS, batch_size=32, shuffle=True)
    model._make_predict_function()
    model.summary()
    print('Loaded new model')

if SAVE_MODEL:
    model.save('model.h5')
    print('Saved new model 2 disk')
    model.summary()
"""
main loop:
we have two threads:
t1 is sending the visual output data via OSC to the JAVA program that is displaying it on the visual object
t2 does the prediction
"""

t1 = threading.Thread(name='ledoutput', target=ledoutput, daemon=True)
t1.daemon = True
t1.start()

SPEC = SpectrumAnalyzer(fft_callback_function, binned=True, send_osc=True)

def loop():
    while 0<1:
        for i in range(NUM_SOUNDS):
            e1.acquire()
        prediction_fft_input = [fft.pop() for x in range(NUM_SOUNDS)]
        prediction_input=np.asarray(prediction_fft_input)
        #prediction_input = np.reshape=(prediction_input, (1,NUM_SOUNDS,INPUT_DIM))
        if is_pause(prediction_fft_input):
            pause_event.set()
            continue
        prediction_input.shape=(1,INPUT_DIM)
        prediction_output=model.predict(prediction_input)
        prediction_output=prediction_output.flatten()
        #print(prediction_input)
        #print("Finished prediction with shape:" + str(prediction_output.shape))
        #print(prediction_output)
        random_value = random.randint(MIN_FRAME_REPLAYS,MAX_FRAME_REPLAYS)
        for i in range(random_value):
            prediction_buffer.append(prediction_output)

t2 = threading.Thread(name='prediction', target=loop, daemon=True)
t2.start()

while True:
    try:
        SPEC.tick()
    except KeyboardInterrupt:
        SPEC.quit()
        sys.exit(0)
