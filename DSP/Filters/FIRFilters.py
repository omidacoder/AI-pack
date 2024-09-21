from scipy.io import wavfile
import numpy as np
import os
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter
import scipy.signal as sg
import matplotlib.pyplot as plt
from scipy.fftpack import fft


def show_filter(b, a):
    w, h = signal.freqs(b, a)
    plt.semilogx(w, 20 * np.log10(abs(h)))
    plt.title('FIR filter frequency response')
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
highcut = 3000
FRAME_RATE = 48000

""" FIR """

# low pass


def fir_lowpass(lowcut, fs):
    low = lowcut 
    b = signal.firwin(501, cutoff=low, fs=fs)
    return b, [1.0]


def fir_lowpass_filter(data, lowcut, fs):
    b, a = fir_lowpass(lowcut, fs)
    y = lfilter(b, a, data)
    return y


def lowpass_filter(buffer):
    return fir_lowpass_filter(buffer, lowcut, FRAME_RATE)

# band pass


def fir_bandpass(lowcut, highcut, fs):
    low = lowcut 
    high = highcut 
    b = firwin(501, [low, high], fs=fs , pass_zero=False)
    return b, [1.0]


def fir_bandpass_filter(data, lowcut, highcut, fs):
    b, a = fir_bandpass(lowcut, highcut, fs)
    y = lfilter(b, a, data)
    return y


def bandpass_filter(buffer):
    return fir_bandpass_filter(buffer, lowcut, highcut, FRAME_RATE)

# high pass


def fir_highpass(highcut, fs):
    high = highcut
    b = firwin(501, cutoff=high , fs=fs , pass_zero=False)
    return b, [1.0]


def fir_highpass_filter(data, highcut, fs):
    b, a = fir_highpass(highcut, fs)
    y = lfilter(b, a, data)
    return y


def highpass_filter(buffer):
    return fir_highpass_filter(buffer, highcut, FRAME_RATE)


# applying firworth filters
filtered = np.apply_along_axis(lowpass_filter, 0, data).astype('int16')
wavfile.write('fir_lowpass.wav', samplerate, filtered)

filtered = np.apply_along_axis(bandpass_filter, 0, data).astype('int16')
wavfile.write('fir_bandpass.wav', samplerate, filtered)

filtered = np.apply_along_axis(highpass_filter, 0, data).astype('int16')
wavfile.write('fir_highpass.wav', samplerate, filtered)
