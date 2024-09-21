from scipy.io import wavfile
import numpy as np
import os
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import cheby2 , lfilter
import scipy.signal as sg
import matplotlib.pyplot as plt
from scipy.fftpack import fft


def show_filter(b, a):
    w, h = signal.freqs(b, a)
    plt.semilogx(w, 20 * np.log10(abs(h)))
    plt.title('Chebyshev 2 filter frequency response')
    plt.xlabel('Frequency [radians / second]')
    plt.ylabel('Amplitude [dB]')
    plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
    plt.axvline(100, color='green')  # cutoff frequency
    plt.show()


def show_signal(audio, title):
    plt.plot(audio)
    plt.ylabel("Amplitude")
    plt.xlabel("Time")
    plt.title(title)
    plt.show()


def fft_plot(audio, sample_rate):
  N = len(audio)    # Number of samples
  T = 1/sample_rate  # Period
  y_freq = fft(audio)
  domain = len(y_freq) // 2
  x_freq = np.linspace(0, sample_rate//2, N//2)
  plt.plot(x_freq, abs(y_freq[:domain]))
  plt.xlabel("Frequency [Hz]")
  plt.ylabel("Frequency Amplitude |X(t)|")
  return plt.show()
# reading file
samplerate, data = wavfile.read('E:\\Darsi\\DSP\\Project\\voice.wav')
show_signal(data, 'original signal')
fft_plot(data, samplerate)

lowcut = 1500
highcut = 5000
FRAME_RATE = 48000
rs = 100 # max ripple count
""" Chebyshev2 """

# low pass

def cheby2_lowpass(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = cheby2(order, rs=rs, Wn=low, btype='lowpass')
    show_filter(b, a)
    return b, a


def cheby2_lowpass_filter(data, lowcut, fs, order=5):
    b, a = cheby2_lowpass(lowcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def lowpass_filter(buffer):
    return cheby2_lowpass_filter(buffer, lowcut, FRAME_RATE, order=6)

# band pass


def cheby2_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = b, a = cheby2(order, rs=rs, Wn=[low,high], btype='bandpass')
    return b, a


def cheby2_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = cheby2_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def bandpass_filter(buffer):
    return cheby2_bandpass_filter(buffer, lowcut, highcut, FRAME_RATE, order=6)

# high pass


def cheby2_highpass(highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a  = cheby2(order, rs=rs, Wn=high, btype='highpass')
    return b, a


def cheby2_highpass_filter(data, highcut, fs, order=5):
    b, a = cheby2_highpass(highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def highpass_filter(buffer):
    return cheby2_highpass_filter(buffer, highcut, FRAME_RATE, order=6)


# applying cheby2worth filters
filtered = np.apply_along_axis(lowpass_filter, 0, data).astype('int16')
wavfile.write('cheby2_lowpass.wav', samplerate, filtered)

filtered = np.apply_along_axis(bandpass_filter, 0, data).astype('int16')
wavfile.write('cheby2_bandpass.wav', samplerate, filtered)

filtered = np.apply_along_axis(highpass_filter, 0, data).astype('int16')
wavfile.write('cheby2_highpass.wav', samplerate, filtered)

