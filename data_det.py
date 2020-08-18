import multiprocessing as mp
import time
import ctypes
import sounddevice as sd
import numpy as np
import sys
import json
# import codecs
import os
import threading
from scipy.io import wavfile
import shutil
from datetime import datetime
import collections
import matlab.engine
from scipy.signal import butter, lfilter, firwin, freqz
import matplotlib.pyplot as plt
import mplcursors
import math

from scipy.io import loadmat

from scipy import signal
from scipy.fft import fftshift
import matplotlib.pyplot as plt

fs = 8000

data = loadmat('D:\Taji\projects\quad_detection\..extracted_data\\14_31_52__5.mat')
data = data['collectedy_DOPT'].real.reshape(-1)

# fs = 10e3
# N = 1e5
# amp = 2 * np.sqrt(2)
# noise_power = 0.01 * fs / 2
# time = np.arange(N) / float(fs)
# mod = 500*np.cos(2*np.pi*0.25*time)
# carrier = amp * np.sin(2*np.pi*3e3*time + mod)
# noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
# noise *= np.exp(-time/5)
# x = carrier + noise

# f, t, Sxx = signal.spectrogram(x=data, fs=fs, window=1024, noverlap=512, nfft=1024)
f, t, Sxx = signal.spectrogram(x=data, fs=fs)
# plt.pcolormesh(t, f, Sxx, shading='gouraud')
plt.pcolormesh(t, f, 20*np.log10(Sxx))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()




