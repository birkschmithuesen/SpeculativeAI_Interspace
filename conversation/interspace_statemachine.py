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
from collections import deque
from pythonosc import udp_client
from pythonosc import osc_server
import numpy as np
from . import fft
from . import neuralnet_audio

LIVE_REPLAY = False # replay the predictions live without buffer

UDP_IP = '127.0.0.1'
UDP_PORT = 10005
OSC_LISTEN_IP = "0.0.0.0" # =>listening from any IP
OSC_LISTEN_PORT = 8000

PAUSE_LENGTH_FOR_RANDOM_ACTIVATION = 550 # length in frames in waiting state triggering random activation
PAUSE_LENGTH = 40 # length in frames of silence that triggers pause event
MINIMUM_MESSAGE_LENGTH  = 6 + PAUSE_LENGTH # ignore all messages below this length
PAUSE_SILENCE_THRESH = 18 # Threshhold defining pause if sum(fft) is below the value
MESSAGE_RANDOMIZER_START = 0 # set the minimum times, how often a frame will be written into the buffer
MESSAGE_RANDOMIZER_END = 0 # set the maximum times, how often a frame will be written into the buffer
PREDICTION_BUFFER_MAXLEN = 440 # 3 seconds * 44.1 fps
UPDATE_FACTOR = 0.5 # factor of how much a ne frame will be multiplied into the prediction buffer. 1 => 100%, 0.5 => 50%
REPLAY_FPS_FACTOR = 5 # qucik & dirty hack to manually adjust the playback speed, because FPS calculation ssems to be wrong

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
    dispatcher.map("/Playback/Recorder/frameCount", neuralnet_audio.frame_count_handler)
    dispatcher.map("/train", neuralnet_audio.train_handler)
    dispatcher.map("/newModel", neuralnet_audio.new_model_handler)
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
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    if not LIVE_REPLAY:
        pause_event.wait()
    while True:
        while len(prediction_buffer) > 0:
            """ Quick & dirty to get the end of the message  black
            """
            time_start = time.time()
            if len(prediction_buffer) < 2:
                #the last values for the LEDs should be black / 0
                prediction_output = prediction_buffer.popleft()[0]
                prediction_output = np.multiply(prediction_output,0)
                prediction_output = prediction_output.astype(np.uint8)
            else:
                prediction_output = prediction_buffer.popleft()[0]
                prediction_output = np.multiply(prediction_output,255)
                prediction_output = prediction_output.astype(np.uint8)
            #prediction_output = prediction_buffer.popleft()[0]
            #prediction_output = np.multiply(prediction_output,255)
            prediction_output = prediction_output.astype(np.uint8)
            for x in range(10):
                ledValues = prediction_output[(x*1402):((x+1)*1402):1]
                ledValues = ledValues - 127;
                header = struct.pack('!IBB',frame_count,x,0)
                message = header+bytes(ledValues.tolist())
                sock.sendto(message, (UDP_IP, UDP_PORT))
            print("Play Frame", len(prediction_buffer))
            #was_talking = True
            time_delta = time.time() - time_start
            if not LIVE_REPLAY:
                sleep_time = (1/(fft.FPS*REPLAY_FPS_FACTOR))-time_delta
                if sleep_time  > 0:
                    time.sleep(sleep_time) #ensure playback speed matches framerate
        #wait till the next frame package is ready
        if not LIVE_REPLAY:
            replay_finished_event.set()
            pause_event.clear()
            pause_event.wait()

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
    print("fft_sum: ", fft_sum)
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
        print("pause_counter: ", pause_counter)
        if pause_counter > PAUSE_LENGTH:
            pause_counter = 0
            return frame_contains_silence, True
    else:
        print("LISTENING!")
        pause_counter = 0
    return frame_contains_silence, False

def add_activation_to_buffer():
    """
    adds random activation into the buffer
    """
    message_length = random.randint(40, 90)
    for i in range(message_length):
        factor = random.randint(0,100)
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
        if activation_counter >= PAUSE_LENGTH_FOR_RANDOM_ACTIVATION:
            add_activation_to_buffer()
            activation_counter = 0
            return InterspaceStateMachine.replaying
        frame_contains_silence, _pause_detected = contains_silence_pause_detected(fft_frame)
        if frame_contains_silence:
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
        global prediction_counter, frames_to_remove, last_frame, should_increase_length
        #frame = 0.7 * [fft_frame] + 0.3 * frame
        frame = np.array([fft_frame]) * UPDATE_FACTOR + last_frame * (1 - UPDATE_FACTOR) #quick & dirty "sliding wwindow solution"
        last_frame = frame
        prediction_input = np.asarray(frame)
        prediction_input.shape = (1, neuralnet_audio.INPUT_DIM)
        prediction_output = neuralnet_audio.model.predict(prediction_input)
        prediction_output = prediction_output.flatten()
        if len(prediction_buffer) < PREDICTION_BUFFER_MAXLEN:
            prediction_counter += 1
        if LIVE_REPLAY:
            random_value = 1
        else:
            random_value = random.randint(
                MESSAGE_RANDOMIZER_START, MESSAGE_RANDOMIZER_END)
            should_increase_length = should_increase_length + random.uniform(-1, 1)
            should_increase_length = np.clip(should_increase_length, -5, 5)
        prediction_buffer.append((prediction_output, prediction_counter))
        if should_increase_length > 0:
            for i in range(random_value):
                prediction_buffer.append((prediction_output, prediction_counter))
        else:
            frames_to_remove += random_value
        while(frames_to_remove > 0):
                 if len(prediction_buffer) > MINIMUM_MESSAGE_LENGTH:
                     prediction_buffer.pop()
                     frames_to_remove -= 1
                 else:
                     break
        print("buffer", len(prediction_buffer))
        print("prediction_counter", prediction_counter)
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
            print(prediction_counter)
            print(len(prediction_buffer))
            prediction_counter = frames_to_remove = activation_counter = 0
            return InterspaceStateMachine.replaying
        else:
            return InterspaceStateMachine.recording

class Replaying(State):
    """
    Replaying the recorded fft frame based predictions and after
    finishing transitioning to waiting state. The replay is done in
    a separate thread triggered by the pause_event threading.Event().
    """
    def run(self, fft_frame):
        if not replay_finished_event.isSet():
            pause_event.set()
    def next(self, fft_frame):
        if replay_finished_event.isSet():
            replay_finished_event.clear()
            print("Transitioned: Waiting")
            return InterspaceStateMachine.waiting
        return InterspaceStateMachine.replaying

class InterspaceStateMachine(StateMachine):
    def __init__(self):
        StateMachine.__init__(self, InterspaceStateMachine.waiting)
        self.t1 = threading.Thread(name='ledoutput', target=ledoutput, daemon=True)
        self.t1.start()
        print("Initialized: Waiting")
        initialize_server()
        neuralnet_audio.run()

spectrum_analyzer = fft.SpectrumAnalyzer(fft_callback_function, binned=True, send_osc=True)
pause_counter = 0
activation_counter = 0
frame_received_semaphore = threading.Semaphore(0)
pause_event = threading.Event()
replay_finished_event = threading.Event()
fft_buffer = []
prediction_buffer = deque(maxlen=PREDICTION_BUFFER_MAXLEN)
frame_count = 0
prediction_counter = 0
frames_to_remove = 0
should_increase_length = 0
last_frame = 0

InterspaceStateMachine.waiting = Waiting()
InterspaceStateMachine.recording = Recording()
InterspaceStateMachine.replaying = Replaying()

if LIVE_REPLAY:
    def new_next_recording(fft_frame):
        return InterspaceStateMachine.recording
    InterspaceStateMachine.recording.next = new_next_recording
