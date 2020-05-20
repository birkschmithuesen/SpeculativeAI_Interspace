"""
This runs the InterspaceStateMachine from statemachine.py
in a thread and runs statemachine.spectrum_analyzer.tick()
in the main thread. This is due to the necessity of pyplot
to run in the main thread and pyplot currently being slow.
"""
import sys
import threading
from conversation import interspace_statemachine

if __name__ == "__main__":
    state_machine = interspace_statemachine.InterspaceStateMachine()

    while True:
        interspace_statemachine.frame_received_semaphore.acquire()
        fft_frame = interspace_statemachine.fft_buffer.pop()
        state_machine.run(fft_frame)
        try:
            interspace_statemachine.spectrum_analyzer.tick()
        except KeyboardInterrupt:
            interspace_statemachine.spectrum_analyzer.quit()
            sys.exit(0)
