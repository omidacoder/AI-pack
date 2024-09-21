# Author : Omid Davar
# Subject : implementing DTFT
# Student Number : 400155017

from scipy.fft import fft, fftfreq
import numpy as np
from matplotlib import pyplot as plt

SAMPLE_RATE = 44100  
DURATION = 5  


def generate_sine_wave(freq, sample_rate, duration):
    x = np.linspace(0, duration, sample_rate * duration, endpoint=False)
    frequencies = x * freq
    # 2pi because np.sin takes radians
    y = np.sin((2 * np.pi) * frequencies)
    return x, y


# Generate a 2 hertz sine wave that lasts for 5 seconds
x, y = generate_sine_wave(2, SAMPLE_RATE, DURATION)
plt.plot(x, y)
plt.show()

# Number of samples in normalized_tone
N = SAMPLE_RATE * DURATION

yf = fft(y)
xf = fftfreq(N, 1 / SAMPLE_RATE)

plt.plot(xf, np.abs(yf))
plt.show()
