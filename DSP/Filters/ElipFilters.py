from scipy.io import wavfile
import numpy as np
import os
from scipy import signal
import matplotlib.pyplot as plt
from scipy.signal import ellip , lfilter
import scipy.signal as sg
import matplotlib.pyplot as plt
from scipy.fftpack import fft

def show_filter(b, a):
    w, h = signal.freqs(b, a)
    plt.semilogx(w, 20 * np.log10(abs(h)))
    plt.title('Elliptical filter frequency response')
    plt.xlabel('Frequency [radians / second]')
    plt.ylabel('Amplitude [dB]')
    # plt.margins(0, 0.1)
    plt.grid(which='both', axis='both')
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
samplerate, data = wavfile.read('voice.wav')
show_signal(data, 'original signal')
fft_plot(data, samplerate)

lowcut = 1500
highcut = 5000
FRAME_RATE = 48000
rs = 36 # max ripple count in stop
rp = 20 # max ripple count in pass
""" Ellips """

# low pass

def ellip_lowpass(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = ellip(order, rs=rs , rp=rp, Wn=low, btype='lowpass')
    show_filter(b, a)
    return b, a


def ellip_lowpass_filter(data, lowcut, fs, order=5):
    b, a = ellip_lowpass(lowcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def lowpass_filter(buffer):
    return ellip_lowpass_filter(buffer, lowcut, FRAME_RATE, order=3)

# band pass


def ellip_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = b, a = ellip(order, rs=rs, rp=rp, Wn=[low, high], btype='bandpass')
    return b, a


def ellip_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = ellip_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def bandpass_filter(buffer):
    return ellip_bandpass_filter(buffer, lowcut, highcut, FRAME_RATE, order=3)

# high pass


def ellip_highpass(highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = ellip(order, rs=rs, rp=rp, Wn=high, btype='highpass')
    return b, a


def ellip_highpass_filter(data, highcut, fs, order=5):
    b, a = ellip_highpass(highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def highpass_filter(buffer):
    return ellip_highpass_filter(buffer, highcut, FRAME_RATE, order=3)


# applying ellipworth filters
filtered = np.apply_along_axis(lowpass_filter, 0, data).astype('int16')
wavfile.write('ellip_lowpass.wav', samplerate, filtered)

filtered = np.apply_along_axis(bandpass_filter, 0, data).astype('int16')
wavfile.write('ellip_bandpass.wav', samplerate, filtered)

filtered = np.apply_along_axis(highpass_filter, 0, data).astype('int16')
wavfile.write('ellip_highpass.wav', samplerate, filtered)

