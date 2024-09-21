# Author : Omid Davar
# Subject : implementing DTFT
# Student Number : 400155017

# Importing packages
import matplotlib.pyplot as plt
import numpy as np
from array import *
import math as math
import cmath as cm


# Defining arrays
f = array('f')
freq = array('i')

# Number of points we are plotting
n = 60

# Defining the delta function
def delta(n):
    if n == 0:
      return 1
    else:
      return 0

# Defining Back Difference function
def backdiff(k):
    return (delta(k)-delta(k-1))/2.0

# Defining Avg Filtering function
def avg(M, val):
    if val in range(0, M):
         temp = 1.0/M
         return temp
    return 0

# Defining DTFT function
def dtft(f, pt):
    output = [0]*n
    for k in range(n):
        s = 0
        p = 0
        for t in range(n):
            s += f[t] * cm.exp(-1j * pt[k] * t)
        output[k] = s
    return output

# Calculating the magnitude of DTFT
def magnitude(inp, n):
    output = [0]*n
    for t in range(0, n):
        tmp = inp[t]
        output[t] = math.sqrt(tmp.real**2 + tmp.imag**2)
    return output

# Calculating the phase
def phase(inp, n):
    output = [0]*n
    for t in range(0, n):
        tmp = inp[t]
        output[t] = math.atan2(tmp.imag, tmp.real)
    return output

# value of the function at n points
avgover = 5
for sec in range(0, n):
    f.append(avg(avgover, sec))

# Defining the x-limits
N = 2*math.pi/n
x = np.arange(-(math.pi), math.pi, N)
x1 = np.fft.fftshift(x)

point = np.arange(n)
freq = np.fft.fftfreq(point.shape[-1])

# Using the function that I made
made_func = dtft(f, x)
made_func_shift = np.fft.fftshift(made_func)
made_func_shift_mag = magnitude(made_func_shift, n)
made_func_shift_phs = phase(made_func_shift, n)

# Using the inbuilt function
inbuilt = np.fft.fft(f, n)
inbuilt_mag = magnitude(inbuilt, n)
inbuilt_phs = phase(inbuilt, n)

# Plotting the Magnitude
plt.figure(1)
plt.subplot(2, 1, 1)
plt.stem(x1, made_func_shift_mag)
plt.xlabel('w')
plt.ylabel('signal')
plt.grid()

# Plotting the Phase angle
plt.subplot(2, 1, 2)
plt.stem(x1, made_func_shift_phs)
plt.xlabel('w')
plt.ylabel('DTFT')
plt.grid()

plt.show()
