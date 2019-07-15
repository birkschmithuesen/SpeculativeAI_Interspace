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
    Predict predict the brightness values of the 13824 Leds of the Interspace object
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
from fft import SpectrumAnalyzer, FPS
import neuralnet_audio

UDP_IP = '127.0.0.1'
UDP_PORT = 10005
OSC_LISTEN_IP = "0.0.0.0" # =>listening from any IP
OSC_LISTEN_PORT = 8000

PAUSE_LENGTH = 9 # length in frames of silence that triggers pause event
PAUSE_SILENCE_THRESH = 10 # Threshhold defining pause if sum(fft) is below the value
MIN_FRAME_REPLAYS = 1 # set the minimum times, how often a frame will be written into the buffer
MAX_FRAME_REPLAYS = 1 # set the maximum times, how often a frame will be written into the buffer
PREDICTION_BUFFER_MAXLEN = 441 # 10 seconds * 44.1 fps

def fft_callback_function(fft_data):
    """
    this function is called when fft values are received via OSC (from ableton Live)
    """
    fft.append(list(fft_data))
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
                header=struct.pack('!IBB',frame_count,x,0)
                message=header+bytes(ledValues.tolist())
                sock.sendto(message, (UDP_IP, UDP_PORT))
            print("Play Frame", len(prediction_buffer))
            #was_talking = True
            time.sleep(1/FPS) #ensure playback speed matches framerate
        #wait till the next frame package is ready
        replay_finished_event.set()
        pause_event.clear()
        pause_event.wait()

def contains_silence(fft_frame):
    """
    Returns true if sum(fft_data) > PAUSE_SILENCE_THRESH
    false otherwise
    """
    timestamp = spectrum_analyzer.last_frame_timestamp
    fft_sum = math.fsum(fft_frame)
    print("fft_sum: ", fft_sum)
    #spectrum_analyzer.log_fft(timestamp, fft_data[0])
    #spectrum_analyzer.log_entry(timestamp, {"sum": fft_sum})
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
        frame_contains_silence, _pause_detected = contains_silence_pause_detected(fft_frame)
        if frame_contains_silence:
            return InterspaceStateMachine.waiting
        print("Transitioned: Recording")
        return InterspaceStateMachine.recording

class Recording(State):
    """
    Recording the fft frames and waiting for detecting a pause
    to transition to replay state
    """
    def run(self, fft_frame):
        frame = [fft_frame]
        prediction_input = np.asarray(frame)
        prediction_input.shape = (1, neuralnet_audio.INPUT_DIM)
        prediction_output = neuralnet_audio.model.predict(prediction_input)
        prediction_output = prediction_output.flatten()
        random_value = random.randint(MIN_FRAME_REPLAYS,MAX_FRAME_REPLAYS)
        for i in range(random_value):
            prediction_buffer.append(prediction_output)
        print("buffer", len(prediction_buffer))
    def next(self, fft_frame):
        _frame_contains_silence, pause_detected = contains_silence_pause_detected(fft_frame)
        if pause_detected:
            print("Transitioned: Replaying")
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

spectrum_analyzer = SpectrumAnalyzer(fft_callback_function, binned=True, send_osc=True)
pause_counter = 0
frame_received_semaphore = threading.Semaphore(0)
pause_event = threading.Event()
replay_finished_event = threading.Event()
fft = []
prediction_buffer = deque(maxlen=PREDICTION_BUFFER_MAXLEN)
frame_count = 0

InterspaceStateMachine.waiting = Waiting()
InterspaceStateMachine.recording = Recording()
InterspaceStateMachine.replaying = Replaying()
