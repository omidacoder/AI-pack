# Author : Omid Davar
# Subject : implementing DFT and IDFT
# Student Number : 400155017
import matplotlib.pyplot as plt
import numpy as np
# defining a function to calculate dft
def DFT(signal):
    N = len(signal)
    n = np.arange(N)
    k = n.reshape((N, 1))
    e = np.exp(-2j * np.pi * k * n / N)
    fourier = np.dot(e, signal)
    return fourier

sampling_rate = 100
signal_range = np.pi/sampling_rate
t = np.arange(0,np.pi,signal_range)

frequency = 1.
signal = 3*np.sin(2*np.pi*frequency*t)

frequency = 4
signal += np.sin(2*np.pi*frequency*t)

frequency = 7   
signal += 0.5* np.sin(2*np.pi*frequency*t)

plt.figure(figsize = (8, 6))
plt.plot(t, signal, 'r')
plt.ylabel('y')
plt.xlabel('signal')
plt.show()

# calculating dft of signal
fourier = DFT(signal)
# calculate the frequencyuency
N = len(fourier)
n = np.arange(N)
T = N/sampling_rate
frequency = n/T 

plt.figure(figsize = (8, 6))
plt.stem(frequency, abs(fourier), 'b', \
         markerfmt=" ", basefmt="-b")
plt.title('DFT Of Signal')
plt.show()

# calculating IDFT
def IDFT(fourier):
    return np.fft.ifft(fourier)
inversed = IDFT(fourier)
plt.figure(figsize = (8, 6))
plt.title('IDFT Result :')
plt.plot(t, inversed, 'r')
plt.ylabel('y')
plt.xlabel('signal')
plt.show()