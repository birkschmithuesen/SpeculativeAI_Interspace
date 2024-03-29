"""
This class contains a state machine model implementing
the Interspace side of message receiving, processing and
sending. It listens to audio makes a dftt transform and makes predictions
utilizing a neural net to visualize on the object Interspace.

The computation process is structured in the following way:
1. when receiving the the OSC command "/train" the function neuralnet_audio.train_handler executes model.fit
2. input: receive audio
3. FFT analysis: separate continues audio stream into 30 frequency bands with a frequency of 44fps
4. prediction: convert the FFT analysis results to feed into a neural network.
   This neural net predicts the brightness values of the 13824 Leds of the Interspace object
"""

import random
import threading
import argparse
import socket
import struct
import math
import time
import datetime
from collections import deque
from pythonosc import udp_client
from pythonosc import osc_server
import numpy as np
from conversation import fft, neuralnet_audio, interspace_artnet

LIVE_REPLAY = False # replay the predictions live without buffer

UDP_IP = '127.0.0.1'
UDP_PORT = 10005
OSC_LISTEN_IP = "0.0.0.0" # =>listening from any IP
OSC_LISTEN_PORT = 8000

PAUSE_LENGTH_FOR_RANDOM_ACTIVATION = 500 # length in frames in waiting state triggering random activation
MINIMUM_MESSAGE_LENGTH  = 8 # ignore all messages below this length
PAUSE_LENGTH = 8 # length in frames of silence that triggers pause event
PAUSE_SILENCE_THRESH = 25 # Threshhold defining pause if sum(fft) is below the value
MESSAGE_RANDOMIZER_START = 0 # set the minimum times, how often a frame will be written into the buffer
MESSAGE_RANDOMIZER_END = 0 # set the maximum times, how often a frame will be written into the buffer
VOLUME_RANDOMIZER_START = 0 # set the minimum value, how much the volume of the different synths will be changed by chance
VOLUME_RANDOMIZER_END = 0 # set the maximum value, how much the volume of the different synths will be changed by chance
PREDICTION_BUFFER_MAXLEN = 4410 # 10 seconds * 44.1 fps
TRACE = 1 #sets the factor how much the actual frame will be mixed with the last one before sendin to Artnet Output. This is the same on the output side as done in the FFT Class with the UPDATE_FACTOR
SPEED_BOOST = 15 #fastens the playback speed. Is needed in SAI communication, because messages tend to get longer, because of fade outs and stuff
MAX_BRIGHTNESS = 255 #sets the maximum brightness. All values will be clamped there. Physical maximum is 255

def fft_callback_function(fft_data):
    """
    this function is called when fft values are received via OSC (from ableton Live)
    """
    fft_buffer.append(list(fft_data))
    frame_received_semaphore.release()

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
    dispatcher.map("/train", neuralnet_audio.continue_training)
    dispatcher.map("/save_model", neuralnet_audio.save_model)

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
    when a new prediction is ready, it sends the LED data via ArtNet over Udp to 'Intersapce'
    """
    global last_frame
    frames = [x[0] for x in prediction_buffer]
    #last_frame = last_frame.astype(int)
    counter = 0
    start_time = datetime.datetime.now()
    for frame in frames:

        #conversion to INT with list
        #int_frame = [int(x * 255) for x in frame]
        #conversion to INT with numpy
        int_frame = np.array(frame)*255
        int_frame = int_frame * TRACE + last_frame *(1 - TRACE)
        last_frame = int_frame
        int_frame = int_frame.astype(int)
        int_frame = int_frame.clip(0, MAX_BRIGHTNESS)
        artnet_sender.send_brightness_buffer(int_frame)
        time_diff = datetime.datetime.now() - start_time
        time_diff = time_diff.total_seconds()
        print("LIGHTING ", counter, "/", len(frames))
        counter = counter+1
        if not LIVE_REPLAY:
            sleep_time = 1.0/(fft.FPS + SPEED_BOOST) - time_diff
            if sleep_time > 0:
                time.sleep(sleep_time) #ensure playback speed matches framerate
        start_time = datetime.datetime.now()
    if not LIVE_REPLAY:
        artnet_sender.all_off()
        time.sleep(0.1)
        artnet_sender.all_off()
    prediction_buffer.clear()

def prediction_buffer_remove_pause():
    """
    Removes dark pause frames at the end of
    prediction_buffer
    """
    global prediction_counter
    # -1 because the last pause frame wrecordon't be recorded in state machine
    last_frame_counter = prediction_counter - (PAUSE_LENGTH - 1)
    if len(prediction_buffer) == 0:
        return
    while(prediction_buffer[-1][1] > last_frame_counter):
        prediction_buffer.pop()
        prediction_counter -= 1
        if len(prediction_buffer) == 0:
           return

def contains_silence(fft_frame):
    """
    Returns true if sum(fft_data) > PAUSE_SILENCE_THRESH
    false otherwise
    """
    timestamp = spectrum_analyzer.last_frame_timestamp
    fft_sum = math.fsum(fft_frame)
    print("level: {:03.0f}".format(fft_sum)," / ",PAUSE_SILENCE_THRESH,"\n")
    return fft_sum < PAUSE_SILENCE_THRESH

def contains_silence_pause_detected(fft_frame):
    """
    returns tuple first one being True if a pause in the fft stream
    was detected else return False. The second one is True if the
    frame contains silence (sound below threshold) or False otherwise.
    """
    global pause_counter
    frame_contains_silence = contains_silence(fft_frame)
    if frame_contains_silence:
        pause_counter += 1
        print("WAITING... (", pause_counter,"/",PAUSE_LENGTH,")")
        if pause_counter > PAUSE_LENGTH:
            pause_counter = 0
            return frame_contains_silence, True
    else:
        #print("LISTENING!")
        pause_counter = 0
    return frame_contains_silence, False

def soundvector_postprocessing(prediction_vector):
    """
    adds some random noise or any other function to the sound vector,
    to add purpose to the answer
    """
    prediction_vector[0] = prediction_vector[0] + random.uniform(VOLUME_RANDOMIZER_START, VOLUME_RANDOMIZER_END)
    prediction_vector[6] = prediction_vector[6] + random.uniform(VOLUME_RANDOMIZER_START, VOLUME_RANDOMIZER_END)
    return prediction_vector

def add_random_activation_to_buffer():
    """
    adds random activation into the buffer
    """
    message_length = random.randint(10, 40)
    for i in range(message_length):
        factor = random.randint(0,70)
        factor = factor/100.0
        message = [np.random.rand(neuralnet_audio.OUTPUT_DIM) * factor]
        prediction_buffer.append(message)

class State:
    def run(self):
        assert 0, "run not implemented"
    def next(self, input):
        assert 0, "next not implemented"

class StateMachine:
    def __init__(self, initialState):
        self.currentState = initialState
        self.currentState.run()

    def run(self, input):
        self.currentState = self.currentState.next(input)
        self.currentState.run(input)

class Waiting(State):
    """
    Waiting for a non silent frame to transition
    to recording state
    """
    def run(self, input=None):
        pass
    def next(self, fft_frame):
        global activation_counter
        if activation_counter >= PAUSE_LENGTH_FOR_RANDOM_ACTIVATION and not LIVE_REPLAY:
            add_random_activation_to_buffer()
            activation_counter = 0
            print("Transitioned: Replaying")
            return InterspaceStateMachine.replaying
        frame_contains_silence, _pause_detected = contains_silence_pause_detected(fft_frame)
        if frame_contains_silence and not LIVE_REPLAY:
            activation_counter += 1
            return InterspaceStateMachine.waiting
        activation_counter = 0
        print("Transitioned: Recording")
        return InterspaceStateMachine.recording

class Recording(State):
    """
    Recording the fft frames and waiting for detecting a pause
    to transition to replay state
    """
    def run(self, fft_frame):
        global prediction_counter, frames_to_remove
        frame = [fft_frame]
        prediction_input = np.asarray(frame)
        prediction_input.shape = (1, neuralnet_audio.INPUT_DIM)
        prediction_output = neuralnet_audio.model.predict(prediction_input)
        prediction_output = prediction_output.flatten()
        prediction_output = soundvector_postprocessing(prediction_output)
        if len(prediction_buffer) < PREDICTION_BUFFER_MAXLEN:
            prediction_counter += 1
        if LIVE_REPLAY:
            random_value = 1
            should_increase_length = 0
        else:
            random_value = random.randint(
                MESSAGE_RANDOMIZER_START, MESSAGE_RANDOMIZER_END)
            should_increase_length = random.randint(
                0, 1)
        prediction_buffer.append((prediction_output, prediction_counter))
        if should_increase_length:
            for i in range(random_value):
                prediction_buffer.append((prediction_output, prediction_counter))
        else:
            frames_to_remove += random_value - 1
        while(frames_to_remove > 0):
                 if len(prediction_buffer) > MINIMUM_MESSAGE_LENGTH:
                     prediction_buffer.pop()
                     frames_to_remove -= 1
                 else:
                     break
        #print("buffer", len(prediction_buffer))
    def next(self, fft_frame):
        global prediction_counter, activation_counter
        _frame_contains_silence, pause_detected = contains_silence_pause_detected(fft_frame)
        activation_counter += 1
        if pause_detected:
            if prediction_counter < MINIMUM_MESSAGE_LENGTH:
                 print("Transitioned: Waiting")
                 prediction_buffer.clear()
                 prediction_counter = frames_to_remove = 0
                 return InterspaceStateMachine.waiting
            print("Transitioned: Replaying")
            prediction_buffer_remove_pause()
            prediction_counter = frames_to_remove = activation_counter = 0
            return InterspaceStateMachine.replaying
        else:
            print("LISTENING!!! (",len(prediction_buffer),")")
            return InterspaceStateMachine.recording

class Replaying(State):
    """
    Replaying the recorded fft frame based predictions and after
    finishing transitioning to waiting state. The replay is done in
    a separate thread triggered by the pause_event threading.Event().
    """
    def run(self, fft_frame):
        ledoutput()
        #if not replay_finished_event.isSet():
        #    pause_event.set()
    def next(self, fft_frame):
        #if replay_finished_event.isSet():
        #    replay_finished_event.clear()
        #    print("Transitioned: Waiting")
        #    return InterspaceStateMachine.waiting
        #return InterspaceStateMachine.replaying
        # we need to empty the buffer and reset the semaphore counter
        # so the chunks buffered during replay will be discarded
        for i in range(len(fft_buffer)):
            frame_received_semaphore.acquire()
            fft_buffer.pop()
        return InterspaceStateMachine.waiting
class InterspaceStateMachine(StateMachine):
    def __init__(self):
        StateMachine.__init__(self, InterspaceStateMachine.waiting)
        #self.t1 = threading.Thread(name='ledoutput', target=ledoutput, daemon=True)
        #self.t1.start()
        print("Initialized: Waiting")
        initialize_server()
        neuralnet_audio.run()

fft_buffer = []
frame_received_semaphore = threading.Semaphore(0)
spectrum_analyzer = fft.SpectrumAnalyzer(fft_callback_function, binned=True, send_osc=True)
artnet_sender = interspace_artnet.InterspaceArtnet()
pause_counter = 0
activation_counter = 0
pause_event = threading.Event()
replay_finished_event = threading.Event()
prediction_buffer = deque(maxlen=PREDICTION_BUFFER_MAXLEN)
frame_count = 0
prediction_counter = 0
frames_to_remove = 0
last_frame = 0

InterspaceStateMachine.waiting = Waiting()
InterspaceStateMachine.recording = Recording()
InterspaceStateMachine.replaying = Replaying()

if LIVE_REPLAY:
    print("Activated LIVE_REPLAY mode")
    def new_next_recording(fft_frame):
        ledoutput()
        return InterspaceStateMachine.recording
    InterspaceStateMachine.recording.next = new_next_recording
