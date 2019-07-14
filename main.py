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
import os
import math

os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #force Tensorflow to use the computed
# the ip and port to send the LED data to. The program Ortlicht receives them via OSC and
# converts them to ArtNet
UDP_IP = '127.0.0.1'
UDP_PORT = 10005
OSC_LISTEN_IP = "0.0.0.0" # =>listening from any IP
OSC_LISTEN_PORT = 8000

LOAD_MODEL = True
SAVE_MODEL = False

model = Sequential()

PREDICTION_BUFFER_MAXLEN = 441 # 10 seconds * 44.1 fps
PAUSE_LENGTH = 9 # length in frames of silence that triggers pause event
PAUSE_SILENCE_THRESH = 10 # Threshhold defining pause if sum(fft) is below the value
MIN_FRAME_REPLAYS = 1 # set the minimum times, how often a frame will be written into the buffer
MAX_FRAME_REPLAYS = 1 # set the maximum times, how often a frame will be written into the buffer

INPUT_DIM = 128
NUM_SOUNDS = 1
BATCH_SIZE = 32
EPOCHS = 30
INITIAL_EPOCHS = 150

HIDDEN1_DIM = 512
HIDDEN2_DIM = 4096
OUTPUT_DIM = 13824


was_talking = threading.Event() #stores the last action True -> Talking; False -> Listening
was_talking.set()
frame_has_sound = threading.Event()
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
    model.add(Dense(OUTPUT_DIM, activation='sigmoid',kernel_initializer=my_init,
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
    #global was_talking
    sock=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pause_event.wait()
    while True:
        while len(prediction_buffer) > 0:
            if len(prediction_buffer) < 2:
                #the last values for the LEDs should be black / 0
                prediction_output = prediction_buffer.popleft()
                prediction_output=np.multiply(prediction_output,0)
                prediction_output=prediction_output.astype(np.uint8)
            else:
                prediction_output = prediction_buffer.popleft()
                prediction_output=np.multiply(prediction_output,255)
                prediction_output=prediction_output.astype(np.uint8)
            for x in range(10):
                ledValues=prediction_output[(x*1402):((x+1)*1402):1]
                header=struct.pack('!IBB',frameCount,x,0)
                message=header+bytes(ledValues.tolist())
                sock.sendto(message, (UDP_IP, UDP_PORT))
            print("Play Frame", len(prediction_buffer))
            #was_talking = True
            time.sleep(1/FPS) #ensure playback speed matches framerate
        #wait till the next frame package is ready
        pause_event.clear()
        pause_event.wait()

def is_silence(fft_data):
    """
    Returns true if sum(fft_data) > PAUSE_SILENCE_THRESH
    false otherwise
    """
    timestamp = SPEC.last_frame_timestamp
    fft_sum = 0
    for fft_frame in fft_data:
        fft_sum += math.fsum(fft_frame)
    print("fft_sum: ", fft_sum)
    #SPEC.log_fft(timestamp, fft_data[0])
    #SPEC.log_entry(timestamp, {"sum": fft_sum})
    return fft_sum < PAUSE_SILENCE_THRESH


def is_pause(fft_data):
    """
    return True if a pause in the fft stream was detected else return False
    """
    global pause_counter
    global was_talking
    global frame_has_sound
    if is_silence(fft_data):
        frame_has_sound.clear()
        pause_counter += 1
        print("Pause_detected: ", pause_counter)
        print("Was Talking:", was_talking.isSet())
        if was_talking.isSet(): the_pause_length = 4 * PAUSE_LENGTH
        else: the_pause_length = PAUSE_LENGTH
        if pause_counter >= the_pause_length:
            pause_counter = 0
            return True
    else:
        frame_has_sound.set()
        print("LISTENING!")
        was_talking.clear()
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
    training_input, training_output = values[:,:-OUTPUT_DIM], values[:,INPUT_DIM:]
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
    global was_talking
    global frame_has_sound
    while 0<1:
        for i in range(NUM_SOUNDS):
            e1.acquire()
        prediction_fft_input = [fft.pop() for x in range(NUM_SOUNDS)]
        if pause_event.isSet():
            continue
        prediction_input=np.asarray(prediction_fft_input)
        if is_pause(prediction_fft_input):
            #remove pause from buffer
            print("REPLAY!!!!");
            was_talking.set()
            #print("was talking", was_talking)
            #for i in range(PAUSE_LENGTH-1):
                #prediction_buffer.pop()
                #prediction_buffer.pop(0)

            pause_event.set()
            continue
        if frame_has_sound.isSet():
            prediction_input.shape=(1,INPUT_DIM)
            prediction_output=model.predict(prediction_input)
            prediction_output=prediction_output.flatten()
            random_value = random.randint(MIN_FRAME_REPLAYS,MAX_FRAME_REPLAYS)
            for i in range(random_value):
                prediction_buffer.append(prediction_output)
            print("buffer", len(prediction_buffer))
        else:
            time.sleep(0.02)

t2 = threading.Thread(name='prediction', target=loop, daemon=True)
t2.start()

while True:
    try:
        SPEC.tick()
    except KeyboardInterrupt:
        SPEC.quit()
        sys.exit(0)
