"""
This module implements an binned fft spectrum analyzer
"""

from threading import Semaphore
import numpy as np
import pyaudio
import matplotlib.pyplot as plt

FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
CHUNK = 2**13
START = 0
N = 2**13
BINS = [(0, 49), (50, 199), (200, 499), (500, 1999),
        (2000, 9999), (10000, 14999), (15000, 20000)]
LOCK = Semaphore(0)

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

    wave_x = 0
    wave_y = 0
    spec_x = 0
    spec_y = 0
    data = []

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
        # Main loop
        plt.ion()
        self.loop()

    def stream_callback(self, in_data, frame_count, time_info, status_flags):
        """
        callback function for PyAudio stream
        """
        LOCK.release()
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
                self.graphplot()
        except KeyboardInterrupt:
            self.stream.close()

        print("End...")

    def fft(self):
        """
        apply fft to audio in current buffer
        """
        self.wave_x = range(START, START + N)
        self.wave_y = self.data[START:START + N]
        self.spec_x = np.fft.rfftfreq(N, d=1.0/RATE)
        spec_y_raw = np.fft.rfft(self.data[START:START + N])
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
        #Spectrum
        plt.subplot(312)
        plt.plot(self.spec_x, self.spec_y, marker='o', linestyle='-')
        plt.axis([0, RATE / 2, 0, 50])
        plt.xlabel("frequency [Hz]")
        plt.ylabel("amplitude spectrum")
        #Bins
        if self.binned:
            plt.subplot(313)
            plt.plot(self.fft_bins_x, self.fft_bins_y, marker='o', linestyle='-')
            #plt.axis([0, RATE / 2, 0, 50])
            plt.xlabel("frequency [Hz]")
            plt.ylabel("amplitude spectrum")
        #Pause
        plt.pause(.02)

if __name__ == "__main__":
    SPEC = SpectrumAnalyzer(binned=True)
