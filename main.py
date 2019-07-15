"""
This runs the InterspaceStateMachine from statemachine.py
in a thread and runs statemachine.spectrum_analyzer.tick()
in the main thread. This is due to the necessity of pyplot
to run in the main thread and pyplot currently being slow.
"""
import sys
import threading
import statemachine
from statemachine import InterspaceStateMachine

# the ip and port to send the LED data to. The program Ortlicht receives them via OSC and
# converts them to ArtNet
statemachine.UDP_PORT = 10005
statemachine.OSC_LISTEN_IP = "0.0.0.0" # =>listening from any IP
statemachine.OSC_LISTEN_PORT = 8000

statemachine.LOAD_MODEL = True
statemachine.SAVE_MODEL = False

statemachine.PAUSE_LENGTH = 9 # length in frames of silence that triggers pause event
statemachine.PAUSE_SILENCE_THRESH = 10 # Threshhold defining pause if sum(fft) is below the value
statemachine.MIN_FRAME_REPLAYS = 1 # set the minimum times, how often a frame will be written into the buffer
statemachine.MAX_FRAME_REPLAYS = 1 # set the maximum times, how often a frame will be written into the buffer
statemachine.PREDICTION_BUFFER_MAXLEN = 441 # 10 seconds * 44.1 fps

statemachine.INPUT_DIM = 128
statemachine.BATCH_SIZE = 32
statemachine.EPOCHS = 30
statemachine.INITIAL_EPOCHS = 150

statemachine.HIDDEN1_DIM = 512
statemachine.HIDDEN2_DIM = 4096
statemachine.OUTPUT_DIM = 13824

if __name__ == "__main__":
    state_machine = InterspaceStateMachine()

    def state_machine_loop():
        while True:
            statemachine.frame_received_semaphore.acquire()
            fft_frame = statemachine.fft.pop()
            state_machine.run(fft_frame)

    t2 = threading.Thread(name='state_machine_loop', target=state_machine_loop, daemon=True)
    t2.start()

    while True:
        try:
            statemachine.spectrum_analyzer.tick()
        except KeyboardInterrupt:
            statemachine.spectrum_analyzer.quit()
            sys.exit(0)
