"""
This module implements an binned fft spectrum analyzer
"""

from threading import Semaphore
import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import time
import sys

DEBUG = True
SHOW_GRAPH = True
FORMAT = pyaudio.paFloat32
FPS = 30
CHANNELS = 1
RATE = 44100
CHUNK = 2**10
START = 0
N = 2**10
WINDOW = np.hanning(N)
BINS = [(a, a+665) for a in range(0, 19980, 666)]
LOCK = Semaphore(0)

def debug(str):
    if DEBUG:
        print(str)

def closest_value_index(val, lst):
    """
    return index of value in list that's closest to given value
    """
    index = 0
    for item in lst:
        if item > val:
            return index
        index += 1
    return index-1

class SpectrumAnalyzer:
    """
    This class contains a spectrum analyzer that optionally plots the results
    """

    def __init__(self, binned=False):
        self.pyaudio = pyaudio.PyAudio()
        self.stream = self.pyaudio.open(format=FORMAT,
                                        channels=CHANNELS,
                                        rate=RATE,
                                        input=True,
                                        output=False,
                                        frames_per_buffer=CHUNK,
                                        stream_callback=self.stream_callback)
        self.binned = binned
        self.fft_bins_x = [str(x) for x in BINS]
        self.fft_bins_y = np.arange(len(BINS))
        self.wave_x = 0
        self.wave_y = 0
        self.spec_x = 0
        self.spec_y = 0
        self.data = []
        self.last_frame_timestamp = time.time()
        self.frame_time_interval = 1.0/FPS
        # Main loop
        plt.ion()
        self.loop()

    def stream_callback(self, in_data, frame_count, time_info, status_flags):
        """
        callback function for PyAudio stream
        """
        if self.time_for_next_frame():
            LOCK.release()
            time_delta = time.time() - self.last_frame_timestamp
            fps = 1.0 / time_delta
            self.last_frame_timestamp = time.time()
            sys.stdout.write("\r{} FPS".format(int(fps)))
            sys.stdout.flush()
            self.data = np.frombuffer(in_data, dtype=np.float32)
            if self.binned:
                self.binned_fft()
            else:
                self.fft()

        return (None, pyaudio.paContinue)

    def loop(self):
        """
        runs loop that plots fft audio results
        """
        try:
            while True:
                if SHOW_GRAPH:
                    self.graphplot()
                else:
                    continue
        except KeyboardInterrupt:
            self.stream.close()

        debug("End...")

    def fft(self):
        """
        apply fft to audio in current buffer
        """
        fft_start_time = time.time()
        self.wave_x = range(START, START + N)
        self.wave_y = self.data[START:START + N]
        self.spec_x = np.fft.rfftfreq(N, d=1.0/RATE)
        windowed_signal = self.data[START:START + N] * WINDOW
        spec_y_raw = np.fft.rfft(windowed_signal)
        self.spec_y = [np.sqrt(c.real ** 2 + c.imag ** 2) for c in spec_y_raw]

    def binned_fft(self):
        """
        calculate fft and return binned results
        """
        self.fft()
        self.fft_bins_y = self.binn_fft()

    def binn_fft(self):
        """
        apply binning to current fft result and return the bins
        """
        bin_res = []
        for fft_bin in BINS:
            bin_res.append(self.bin_spec_y(fft_bin[0], fft_bin[1]))
        return bin_res

    def bin_spec_y(self, start, end):
        """
        calculate value of single spectrum bin from start frequency
        to end frequency and return the average of the bin frequencies
        """
        #print(self.spec_x.tolist())
        start_spec_x = closest_value_index(start, self.spec_x.tolist())
        i = 0
        bin_sum = 0
        while(start_spec_x + i < len(self.spec_x) and self.spec_x[start_spec_x + i] <= end):
            bin_sum += self.spec_y[start_spec_x + i]
            i += 1
        average = bin_sum / (i+1)
        return average

    def time_for_next_frame(self):
        return time.time() > self.last_frame_timestamp + self.frame_time_interval

    def graphplot(self):
        """
        draw graph of audio and fft data
        """
        LOCK.acquire()
        plt.clf()
        # wave
        plt.subplot(311)
        plt.plot(self.wave_x, self.wave_y)
        plt.axis([START, START + N, -0.5, 0.5])
        plt.xlabel("time [sample]")
        plt.ylabel("amplitude")
        # spectrum
        plt.subplot(312)
        plt.axis([0, RATE / 2, 0, 50])
        plt.xlabel("frequency [Hz]")
        plt.ylabel("amplitude spectrum")
        if self.binned:
            plt.plot(self.spec_x, self.fft_bins_y, marker='o', linestyle='-')
        else:
            plt.plot(self.spec_x, self.spec_y, marker='o', linestyle='-')
        plt.pause(.001)

if __name__ == "__main__":
    SPEC = SpectrumAnalyzer(binned=False)
