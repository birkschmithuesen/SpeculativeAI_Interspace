"""
This module implements an binned fft spectrum analyzer
"""

import os
import numpy as np
import pyaudio
import time
import sys
from pythonosc import osc_message_builder
from pythonosc import udp_client
import math
import csv

def createBins():
    """
    creates the BIN Array for the FFT. The function creates bigger intervals for higher frequencies
    """
    theBins = []
    startFreq = 50
    for a in range(32):
        endFreq = int(startFreq*1.15+10)
        theRange = (startFreq, endFreq)
        startFreq = endFreq
        theBins.append(theRange)
    return(theBins)

DEBUG = False
SHOW_GRAPH = True
FPS = 44.1
UPDATE_FACTOR = 0.5 # factor of how much a ne frame will be multiplied into the prediction buffer. 1 => 100%, 0.5 => 50%

OSC_IP = "127.0.0.1"
OSC_PORT = 8001

FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100
CHUNK = 1024 # int(RATE/FPS)
#FPS = RATE/CHUNK # ca 44.1
START = 0
N = CHUNK
WINDOW = np.hanning(N)
BINS = createBins() #[(a, a+198) for a in range(30, 6000, 199)]
#BINS = [(a, a+82) for a in range(20, 10020, 83)]

if SHOW_GRAPH:
    import matplotlib.pyplot as plt

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

    def __init__(self, fft_function, binned=True, send_osc=True):
        self.fft_callback = fft_function
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
        self.send_osc = send_osc
        if self.send_osc:
            self.client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
        # Main loop
        if SHOW_GRAPH:
            plt.ion()
        if DEBUG:
            self.log = dict()

    def stream_callback(self, in_data, frame_count, time_info, status_flags):
        """
        callback function for PyAudio stream
        """
        time_delta = time.time() - self.last_frame_timestamp
        if time_delta == 0:
            time_delta = sys.float_info.min
        fps = int(1.0 / time_delta)
        self.last_frame_timestamp = time.time()
        if DEBUG:
            self.log_fps(self.last_frame_timestamp, fps)
            sys.stdout.write("\r{} FPS".format(fps))
            sys.stdout.flush()
        self.data = np.frombuffer(in_data, dtype=np.float32)
        if self.binned:
            self.binned_fft()
            self.fft_callback(self.fft_bins_y)
        else:
            self.fft()
            self.fft_callback(self.spec_y)
        if self.send_osc:
            self.send_fft_osc()
        return (None, pyaudio.paContinue)

    def send_fft_osc(self):
        """
        send the computed fft results via OSC to the set IP and port
        """
        self.client.send_message("/fft_train", list(self.fft_bins_y))

    def tick(self):
        """
        runs loop that plots fft audio results
        """
        try:
            if SHOW_GRAPH:
                self.graphplot()
        except KeyboardInterrupt:
            self.quit()
            print("\nProgram stopped...")

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

    def graphplot(self):
        """
        draw graph of audio and fft data
        """
        plt.clf()
        # wave
        plt.subplot(311)
        plt.plot(self.wave_x, self.wave_y)
        plt.axis([START, START + N, -0.5, 0.5])
        plt.xlabel("time [sample]")
        plt.ylabel("amplitude")
        # spectrum
        plt.subplot(312)
        axes = plt.gca()
        axes.set_ylim([0,30])
        axes.xaxis.set_major_locator(plt.MaxNLocator(4))
        plt.xlabel("frequency [Hz]")
        plt.ylabel("amplitude spectrum")
        if self.binned:
            plt.plot(self.fft_bins_x, self.fft_bins_y, marker='o', linestyle='-')
        else:
            plt.plot(self.spec_x, self.spec_y, marker='o', linestyle='-')
        plt.pause(.01)

    def quit(self):
        """
        Quit the spectrum analyzer, closing audio stream and plots
        as well as writing the debug log depending on settings.
        """
        self.stream.close()
        if SHOW_GRAPH:
            plt.close('all')
        if DEBUG:
            self.write_debug_log()
        sys.exit(0)

    def log_fps(self, timestamp, fps):
        """
        log given fp
        """
        entry = {"fps": fps}
        self.log_entry(timestamp, entry)

    def log_fft(self, timestamp, fft):
        """
        log given fft list
        """
        entry = {}
        for binval, i in zip(list(fft),range(30)):
            name = "fft" + str(i)
            entry.update({name: binval})
        self.log_entry(timestamp, entry)

    def log_entry(self, timestamp, entry):
        """
        log abitrary information contained in an dict.
        """
        if timestamp in self.log:
            self.log[timestamp].update(entry)
        else:
            self.log[timestamp] = entry

    def write_debug_log(self):
        with open("debug_log.csv", mode="w") as csv_file:
            fieldnames = ["timestamp", "fps"]
            fieldnames.extend(["fft" + str(i) for i in range(30)])
            fieldnames.extend(["sum"])
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for key, val in self.log.items():
                val.update({"timestamp": key})
                writer.writerow(val)
            abspath = os.path.realpath(csv_file.name)
            print("\n\nWritten log to {}".format(abspath))

if __name__ == "__main__":
    SPEC = SpectrumAnalyzer(lambda x: x, binned=True, send_osc=True)
    while True:
        SPEC.tick()
