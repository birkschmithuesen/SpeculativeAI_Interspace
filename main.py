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
